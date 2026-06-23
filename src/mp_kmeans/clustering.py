"""
KMeans++ with Multi-Precision Support and Normalization
"""
import math
import time
from typing import Optional, Tuple, Literal

import torch
from . import euclidean_cuda


class KMeansPlusPlus:
    """
    KMeans++ clustering with multiple precision modes and normalization
    
    Normalization options:
    - None: No normalization
    - 'l2': L2 normalization (unit vectors)
    - 'standard': Standardization (zero mean, unit variance)
    - 'minmax': Min-max scaling to [0, 1]

    K-means|| controls:
    - km_parallel_oversampling: oversampling factor l (None -> 2*k)
    - km_parallel_rounds: number of sampling rounds (None -> ceil(log2(N+1)))
    """
    
    def __init__(
        self,
        n_clusters: int,
        kernel: Literal[
            'fp16', 'bf16', 'fp32', 'fp64',
            'fp16_uniform', 'bf16_uniform', 'tf32_uniform',
            'fp16_fp64', 'bf16_fp64', 'tf32_fp64', 'fp32_fp64',
            'fp64_fp16', 'fp64_bf16', 'fp64_tf32', 'fp64_fp32_gemm'
        ] = 'fp32',
        kappa: float = 5.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        normalize: Optional[Literal['l2', 'standard', 'minmax']] = 'standard',
        random_state: Optional[int] = None,
        init_method: Literal['kmeans++', 'random', 'k-means||'] = 'random',
        verbose: bool = False,
        km_parallel_oversampling: Optional[int] = None,
        km_parallel_rounds: Optional[int] = None,
    ):  
        self.init_method = init_method
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.kappa = kappa
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        self.random_state = random_state
        self.verbose = verbose
        self.km_parallel_oversampling = km_parallel_oversampling
        self.km_parallel_rounds = km_parallel_rounds
        
        # Normalization parameters (learned from training data)
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        
        # Results
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.timing_ = {}
        
    def _get_dtype(self):
        """Get appropriate dtype for kernel"""
        dtype_map = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32,
            'fp64': torch.float64,
            'fp16_uniform': torch.float16,
            'bf16_uniform': torch.bfloat16,
            'tf32_uniform': torch.float32,
            'fp16_fp64': torch.float16,
            'bf16_fp64': torch.bfloat16,
            'tf32_fp64': torch.float32,
            'fp32_fp64': torch.float32,
            'fp64_fp16': torch.float16,
            'fp64_bf16': torch.bfloat16,
            'fp64_tf32': torch.float32,
            'fp64_fp32_gemm': torch.float64,
        }
        return dtype_map.get(self.kernel, torch.float32)
    
    def _normalize_data(self, X: torch.Tensor, fit: bool = True) -> torch.Tensor:
        """
        Normalize data according to self.normalize
        
        Args:
            X: Input data (N, D)
            fit: If True, learn normalization parameters from X
        
        Returns:
            Normalized data
        """
        if self.normalize is None:
            return X
        
        
        if self.normalize == 'l2':
            # L2 normalization (unit vectors)
            norms = torch.norm(X, p=2, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-12)  # Avoid division by zero
            X_norm = X / norms
            
        elif self.normalize == 'standard':
            # Standardization (zero mean, unit variance)
            if fit:
                self.mean_ = X.mean(dim=0, keepdim=True)
                self.std_ = X.std(dim=0, keepdim=True)
                self.std_ = torch.clamp(self.std_, min=1e-8)  # Avoid division by zero
            
            X_norm = (X - self.mean_) / self.std_
            
        elif self.normalize == 'minmax':
            # Min-max scaling to [0, 1]
            if fit:
                self.min_ = X.min(dim=0, keepdim=True).values
                self.max_ = X.max(dim=0, keepdim=True).values
                range_ = self.max_ - self.min_
                range_ = torch.clamp(range_, min=1e-8)
                self.range_ = range_
            
            X_norm = (X - self.min_) / self.range_
            
        else:
            raise ValueError(f"Unknown normalization: {self.normalize}")
        
        return X_norm
    
    def _compute_distances(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Compute distances using specified kernel"""
        
        # Ensure correct dtype for uniform CUDA kernels
        if self.kernel == 'fp16_uniform':
            # Convert to FP16 if needed
            if X.dtype != torch.float16:
                X = X.half()
            if centers.dtype != torch.float16:
                centers = centers.half()
            # Output is FP16, convert to FP32 for consistency
            D = euclidean_cuda.pairwise_euclidean_uniform_fp16(X, centers)
            return D.float()  # Convert to FP32 for label assignment
        
        elif self.kernel == 'bf16_uniform':
            if X.dtype != torch.bfloat16:
                X = X.to(torch.bfloat16)
            if centers.dtype != torch.bfloat16:
                centers = centers.to(torch.bfloat16)
            D = euclidean_cuda.pairwise_euclidean_uniform_bf16(X, centers)
            return D.float()
        
        elif self.kernel == 'tf32_uniform':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_uniform_tf32(X, centers)
        
        elif self.kernel == 'fp32_uniform':
            if X.dtype != torch.float32:
                X = X.float()
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_single(X, centers)
        
        elif self.kernel == 'fp64_uniform':
            if X.dtype != torch.float64:
                X = X.double()
                centers = centers.double()
            return euclidean_cuda.pairwise_euclidean_double(X, centers)

        # PyTorch native uniform (original)
        elif self.kernel == 'fp16':
            if X.dtype != torch.float16:
                X = X.half()
            if centers.dtype != torch.float16:
                centers = centers.half()
            D = torch.cdist(X, centers, p=2.0) ** 2
            return D.float()
        
        elif self.kernel == 'bf16':
            if X.dtype != torch.bfloat16:
                X = X.to(torch.bfloat16)
            if centers.dtype != torch.bfloat16:
                centers = centers.to(torch.bfloat16)
            D = torch.cdist(X, centers, p=2.0) ** 2
            return D.float()
        
        elif self.kernel == 'fp32':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return torch.cdist(X, centers, p=2.0) ** 2
        
        elif self.kernel == 'fp64':
            if X.dtype != torch.float64:
                X = X.double()
            if centers.dtype != torch.float64:
                centers = centers.double()
            return torch.cdist(X, centers, p=2.0) ** 2
        
        # Mixed-precision modes
        elif self.kernel == 'fp16_fp64':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_fp16_fp64(X, centers, self.kappa)
        
        elif self.kernel == 'bf16_fp64':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_bf16_fp64(X, centers, self.kappa)
        
        elif self.kernel == 'tf32_fp64':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_tf32_fp64(X, centers, self.kappa)
        
        elif self.kernel == 'fp32_fp64':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_fp32_fp64(X, centers, self.kappa)
        
        # Mixed-precision modes
        elif self.kernel == 'fp16_fp32':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_fp16_fp32(X, centers, self.kappa)
        
        elif self.kernel == 'bf16_fp32':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_bf16_fp32(X, centers, self.kappa)
        
        elif self.kernel == 'fp32_fp32':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_fp32_fp32(X, centers, self.kappa)
        
        # Advanced modes
        elif self.kernel == 'fp64_fp16':
            if X.dtype != torch.float16:
                X = X.half()
            if centers.dtype != torch.float16:
                centers = centers.half()
            return euclidean_cuda.pairwise_euclidean_fp64_fp16(X, centers, self.kappa)
        
        elif self.kernel == 'fp64_bf16':
            if X.dtype != torch.bfloat16:
                X = X.to(torch.bfloat16)
            if centers.dtype != torch.bfloat16:
                centers = centers.to(torch.bfloat16)
            return euclidean_cuda.pairwise_euclidean_fp64_bf16(X, centers, self.kappa)
        
        elif self.kernel == 'fp64_tf32':
            if X.dtype != torch.float32:
                X = X.float()
            if centers.dtype != torch.float32:
                centers = centers.float()
            return euclidean_cuda.pairwise_euclidean_fp64_tf32(X, centers, self.kappa)
        
        elif self.kernel == 'fp64_fp32_gemm':
            if X.dtype != torch.float64:
                X = X.double()
            if centers.dtype != torch.float64:
                centers = centers.double()
            return euclidean_cuda.pairwise_euclidean_fp64_fp32_gemm(X, centers, self.kappa)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    

    def fit(self, X: torch.Tensor, C: torch.Tensor=None) -> 'KMeansPlusPlus':
        """
        Fit KMeans model
        
        Args:
            X: Data tensor (N, D)
        
        Returns:
            self
        """
        if not X.is_cuda:
            X = X.cuda()
        
        # Normalize data (fit normalization parameters)
        t_norm_start = time.time()
        X_normalized = self._normalize_data(X, fit=True)
        t_norm = time.time() - t_norm_start
        
        if self.verbose and self.normalize:
            print(f"Normalization ({self.normalize}): {t_norm:.4f}s")
        
        # Convert to appropriate dtype
        target_dtype = self._get_dtype()
        
        N, D = X_normalized.shape
        
        if C is not None:
            if C.shape != (self.n_clusters, D):
                raise ValueError(f"Provided centers C must have shape ({self.n_clusters}, {D})")
            centers = C.to(X_normalized.device).to(target_dtype)
            self.timing_['init'] = 0.0
        else:
            # Initialize centers
            t0 = time.time()
            centers = self._init_centers(X_normalized)
            self.timing_['init'] = time.time() - t0

        self.timing_['normalization'] = t_norm
        
        # Lloyd's algorithm
        self.timing_['iterations'] = []
        
        if self.kernel == "fp64":
            for iteration in range(self.max_iter):
                iter_start = time.time()
                
                # E-step: Assign points to nearest center
                distances = self._compute_distances(X_normalized.to(target_dtype), centers)
                
                # Handle different output dtypes
                if distances.dtype in [torch.float16, torch.bfloat16]:
                    distances_float = distances.float()
                else:
                    distances_float = distances
                
                labels = distances_float.argmin(dim=1)
                
                # M-step: Update centers using CUDA kernel (FAST!)
                new_centers = euclidean_cuda.update_centers_fp64_with_reinit(
                    X_normalized.double(),  # Convert to FP64 for center computation
                    labels,
                    self.n_clusters
                )
                new_centers = new_centers.to(target_dtype)
                
                # Check convergence
                center_shift = torch.norm(new_centers.float() - centers.float()).item()
                centers = new_centers
                
                iter_time = time.time() - iter_start
                self.timing_['iterations'].append(iter_time)
                
                if self.verbose:
                    print(f"Iteration {iteration+1}: shift={center_shift:.6f}, time={iter_time:.4f}s")
                
                if center_shift < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration+1}")
                    break
            
        else:
            for iteration in range(self.max_iter):
                iter_start = time.time()
                
                # E-step: Assign points to nearest center
                distances = self._compute_distances(X_normalized.to(target_dtype), centers)
                
                # Handle different output dtypes
                if distances.dtype in [torch.float16, torch.bfloat16]:
                    distances_float = distances.float()
                else:
                    distances_float = distances
                
                labels = distances_float.argmin(dim=1)
                
                # M-step: Update centers using CUDA kernel (FAST!)
                new_centers = euclidean_cuda.update_centers_fp32_with_reinit(
                    X_normalized.float(),  # Convert to FP32 for center computation
                    labels,
                    self.n_clusters
                )
                new_centers = new_centers.to(target_dtype)
                
                # Check convergence
                center_shift = torch.norm(new_centers.float() - centers.float()).item()
                centers = new_centers
                
                iter_time = time.time() - iter_start
                self.timing_['iterations'].append(iter_time)
                
                if self.verbose:
                    print(f"Iteration {iteration+1}: shift={center_shift:.6f}, time={iter_time:.4f}s")
                
                if center_shift < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration+1}")
                    break
            
        self.n_iter_ = iteration + 1
        
        # Final assignment
        distances = self._compute_distances(X_normalized.to(target_dtype), centers)
        if distances.dtype in [torch.float16, torch.bfloat16]:
            distances = distances.float()
        
        self.labels_ = distances.argmin(dim=1)
        self.cluster_centers_ = centers  # Centers are in normalized space
        
        # Compute inertia
        min_dists = distances.min(dim=1).values
        self.inertia_ = min_dists.sum().item()
        
        # Total timing
        self.timing_['total'] = sum(self.timing_['iterations']) + self.timing_['init'] + self.timing_['normalization']
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels for X"""
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted yet")
        
        if not X.is_cuda:
            X = X.cuda()
        
        # Normalize using learned parameters
        X_normalized = self._normalize_data(X, fit=False)
        
        target_dtype = self._get_dtype()
        
        distances = self._compute_distances(X_normalized.to(target_dtype), self.cluster_centers_)
        if distances.dtype in [torch.float16, torch.bfloat16]:
            distances = distances.float()
        
        return distances.argmin(dim=1)
    
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_
    
    def get_cluster_centers_original_space(self) -> torch.Tensor:
        """
        Get cluster centers in original (un-normalized) space
        
        Returns:
            Centers in original space
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted yet")
        
        centers = self.cluster_centers_.float()
        
        if self.normalize == 'standard':
            # Inverse standardization
            centers = centers * self.std_ + self.mean_
        elif self.normalize == 'minmax':
            # Inverse min-max scaling
            centers = centers * self.range_ + self.min_
        elif self.normalize == 'l2':
            # Cannot invert L2 normalization uniquely
            import warnings
            warnings.warn("Cannot invert L2 normalization. Returning normalized centers.")
        
        return centers


    def _init_centers(self, X: torch.Tensor) -> torch.Tensor:
        method = self.init_method.lower().replace(" ", "")

        if method == "random":
            return self._init_centers_random(X)
        if method in ("kmeans++", "k-means++"):
            return self._init_centers_kmeanspp(X)
        if method in ("kmeans||", "k-means||", "kmeansparallel"):
            return self._init_centers_kmeans_parallel(X)

        raise ValueError(
            f"Unknown init_method={self.init_method!r}. "
            f"Expected one of: 'random', 'kmeans++', 'kmeans||'."
        )
        
    def _make_generator(self, device: torch.device) -> torch.Generator:
        g = torch.Generator(device=device)
        if self.random_state is not None:
            g.manual_seed(int(self.random_state))
        return g

    def _sq_euclidean_distances_to_point(self, X: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        diff = X - x.unsqueeze(0)
        return (diff * diff).sum(dim=1)

    def _assign_to_centers(self, X: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X32 = X if X.dtype == torch.float32 else X.float()
        C32 = centers if centers.dtype == torch.float32 else centers.float()
        dists = torch.cdist(X32, C32, p=2.0) ** 2
        min_sq_dists, labels = dists.min(dim=1)
        return labels, min_sq_dists

    def _weighted_kmeanspp_on_candidates(
        self,
        candidates: torch.Tensor,
        weights: torch.Tensor,
        n_select: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        M, D = candidates.shape
        device = candidates.device
        dtype = candidates.dtype

        C32 = candidates if candidates.dtype == torch.float32 else candidates.float()
        weights32 = weights if weights.dtype == torch.float32 else weights.float()

        centers = torch.empty(n_select, D, device=device, dtype=dtype)

        total_w = weights32.sum()
        if total_w <= 0:
            first_idx = torch.randint(0, M, (1,), device=device, generator=generator).item()
        else:
            first_idx = torch.multinomial(weights32 / total_w, 1, generator=generator).item()

        centers[0] = candidates[first_idx]

        min_sq_dists = self._sq_euclidean_distances_to_point(C32, C32[first_idx])

        chosen = torch.zeros(M, dtype=torch.bool, device=device)
        chosen[first_idx] = True
        min_sq_dists[chosen] = 0.0

        for k in range(1, n_select):
            sampling_weights = weights32 * min_sq_dists
            sampling_weights[chosen] = 0.0
            total = sampling_weights.sum()

            if total <= 0:
                remaining = (~chosen).nonzero(as_tuple=False).flatten()
                if remaining.numel() == 0:
                    next_idx = torch.randint(0, M, (1,), device=device, generator=generator).item()
                else:
                    pos = torch.randint(0, remaining.numel(), (1,), device=device, generator=generator).item()
                    next_idx = remaining[pos].item()
            else:
                probs = sampling_weights / total
                next_idx = torch.multinomial(probs, 1, generator=generator).item()

            centers[k] = candidates[next_idx]
            chosen[next_idx] = True

            new_sq_dists = self._sq_euclidean_distances_to_point(C32, C32[next_idx])
            min_sq_dists = torch.minimum(min_sq_dists, new_sq_dists)
            min_sq_dists[chosen] = 0.0

        return centers

    def _init_centers_random(self, X: torch.Tensor) -> torch.Tensor:
        N, D = X.shape
        device = X.device
        dtype = X.dtype
        g = self._make_generator(device)

        centers = torch.empty(self.n_clusters, D, device=device, dtype=dtype)

        if self.n_clusters <= N:
            perm = torch.randperm(N, generator=g, device=device)
            chosen_idx = perm[:self.n_clusters]
        else:
            chosen_idx = torch.randint(0, N, (self.n_clusters,), generator=g, device=device)

        centers.copy_(X[chosen_idx])

        if self.verbose:
            print(f"Random initialization (kernel: {self.kernel}, normalize: {self.normalize})...")

        return centers

    def _init_centers_kmeanspp(self, X: torch.Tensor) -> torch.Tensor:
        N, D = X.shape
        device = X.device
        dtype = X.dtype
        g = self._make_generator(device)

        X32 = X if X.dtype == torch.float32 else X.float()
        centers = torch.empty(self.n_clusters, D, device=device, dtype=dtype)

        if self.verbose:
            print(f"KMeans++ initialization (kernel: {self.kernel}, normalize: {self.normalize})...")

        first_idx = torch.randint(0, N, (1,), generator=g, device=device).item()
        centers[0] = X[first_idx]

        chosen = torch.zeros(N, dtype=torch.bool, device=device)
        chosen[first_idx] = True

        min_sq_dists = self._sq_euclidean_distances_to_point(X32, X32[first_idx])
        min_sq_dists[chosen] = 0.0

        for k in range(1, self.n_clusters):
            weights = min_sq_dists.clone()
            weights[chosen] = 0.0
            total = weights.sum()

            if total <= 0:
                remaining = (~chosen).nonzero(as_tuple=False).flatten()
                if remaining.numel() == 0:
                    next_idx = torch.randint(0, N, (1,), generator=g, device=device).item()
                else:
                    pos = torch.randint(0, remaining.numel(), (1,), generator=g, device=device).item()
                    next_idx = remaining[pos].item()
            else:
                probs = weights / total
                next_idx = torch.multinomial(probs, 1, generator=g).item()

            centers[k] = X[next_idx]
            chosen[next_idx] = True

            new_sq_dists = self._sq_euclidean_distances_to_point(X32, X32[next_idx])
            min_sq_dists = torch.minimum(min_sq_dists, new_sq_dists)
            min_sq_dists[chosen] = 0.0

        return centers
        
    def _sq_euclidean_distances_to_centers(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Euclidean distances from each row of X to each center.
        X: (N, D), centers: (M, D)
        Returns: (N, M)
        """
        X32 = X if X.dtype == torch.float32 else X.float()
        C32 = centers if centers.dtype == torch.float32 else centers.float()
        return torch.cdist(X32, C32, p=2.0) ** 2
        
    def _init_centers_kmeans_parallel(self, X: torch.Tensor) -> torch.Tensor:
        """
        Tensorized k-means|| initialization under squared Euclidean distance.
    
        Procedure:
        1. Sample one initial center uniformly.
        2. Run R rounds of parallel D^2-sampling with oversampling factor l.
        3. Assign weights to candidate centers.
        4. Recluster the candidates down to K centers using weighted k-means++ seeding.
        """
        N, D = X.shape
        device = X.device
        dtype = X.dtype
        g = self._make_generator(device)
    
        X32 = X if X.dtype == torch.float32 else X.float()
    
        l = self.km_parallel_oversampling
        if l is None:
            l = max(2 * self.n_clusters, 1)
    
        R = self.km_parallel_rounds
        if R is None:
            R = max(1, int(math.ceil(math.log2(N + 1))))
    
        if self.verbose:
            print(
                f"KMeans|| initialization (kernel: {self.kernel}, normalize: {self.normalize}, "
                f"oversampling={l}, rounds={R})..."
            )
    
        # First center
        first_idx = torch.randint(0, N, (1,), generator=g, device=device)
        candidate_indices = first_idx.clone()   # shape: (1,)
    
        # Initial nearest squared distances to the first center
        min_sq_dists = self._sq_euclidean_distances_to_point(X32, X32[first_idx.item()])
        min_sq_dists[first_idx] = 0.0
    
        # R rounds of parallel D^2-sampling
        for t in range(R):
            phi = min_sq_dists.sum()
    
            if not torch.isfinite(phi):
                raise ValueError("Encountered non-finite potential in k-means|| initialization")
    
            if phi <= 0:
                sampled_indices = torch.empty(0, dtype=torch.long, device=device)
            else:
                probs = (l * min_sq_dists / phi).clamp_(max=1.0)
                sampled_mask = torch.rand(N, device=device, generator=g) < probs
                sampled_indices = torch.nonzero(sampled_mask, as_tuple=False).flatten()
    
            if sampled_indices.numel() > 0:
                # Tensorized candidate accumulation
                candidate_indices = torch.cat([candidate_indices, sampled_indices], dim=0)
    
                # Update min distances using only newly sampled centers
                new_centers = X32[sampled_indices]  # (m, D)
                new_sq_dists = self._sq_euclidean_distances_to_centers(X32, new_centers)  # (N, m)
                new_min_sq_dists = new_sq_dists.min(dim=1).values
                min_sq_dists = torch.minimum(min_sq_dists, new_min_sq_dists)
    
                # already selected candidates have zero distance to candidate set
                min_sq_dists[candidate_indices] = 0.0
    
            if self.verbose:
                print(f"  Round {t + 1}/{R}: sampled {sampled_indices.numel()} candidates")
    
        # Deduplicate candidate indices
        candidate_indices = torch.unique(candidate_indices, sorted=False)
    
        candidates = X[candidate_indices]   # original dtype for returned centers
    
        if candidates.shape[0] == 0:
            # Extremely defensive fallback
            return self._init_centers_random(X)
    
        # Assign weights to candidates
        labels, _ = self._assign_to_centers(X, candidates)
        weights = torch.bincount(labels, minlength=candidates.shape[0]).to(device=device, dtype=torch.float32)
    
        # If candidates are fewer than required centers, pad by random sampling
        if candidates.shape[0] < self.n_clusters:
            if self.verbose:
                print(
                    f"  Only {candidates.shape[0]} unique candidates for K={self.n_clusters}; "
                    f"padding with random points."
                )
    
            extra_needed = self.n_clusters - candidates.shape[0]
            extra_idx = torch.randint(0, N, (extra_needed,), generator=g, device=device)
            extra_centers = X[extra_idx]
            return torch.cat([candidates, extra_centers], dim=0)
    
        # Recluster candidates down to K centers using weighted k-means++ seeding
        final_centers = self._weighted_kmeanspp_on_candidates(
            candidates=candidates,
            weights=weights,
            n_select=self.n_clusters,
            generator=g,
        )
    
        return final_centers

def make_blobs_gpu(
    n_samples: int,
    n_features: int,
    n_centers: int,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    random_state: Optional[int] = None,
    return_centers: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Gaussian blobs on GPU
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_centers: Number of cluster centers
        cluster_std: Standard deviation of clusters
        center_box: Bounding box for centers
        random_state: Random seed
        return_centers: If True, also return true centers
    
    Returns:
        X: Data (n_samples, n_features)
        y: True labels (n_samples,)
        centers: True centers (n_centers, n_features) [if return_centers=True]
    """
    if random_state is not None:
        torch.manual_seed(random_state)
    
    # Generate centers
    centers = torch.rand(n_centers, n_features, device='cuda') * (center_box[1] - center_box[0]) + center_box[0]
    
    # Generate samples
    samples_per_center = n_samples // n_centers
    remainder = n_samples % n_centers
    
    X_list = []
    y_list = []
    
    for i in range(n_centers):
        n = samples_per_center + (1 if i < remainder else 0)
        X_cluster = centers[i:i+1] + torch.randn(n, n_features, device='cuda') * cluster_std
        y_cluster = torch.full((n,), i, device='cuda', dtype=torch.long)
        X_list.append(X_cluster)
        y_list.append(y_cluster)
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Shuffle
    perm = torch.randperm(n_samples, device='cuda')
    X = X[perm]
    y = y[perm]
    
    if return_centers:
        return X, y, centers
    return X, y
