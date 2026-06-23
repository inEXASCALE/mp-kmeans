"""
Comprehensive KMeans precision mode comparison with sklearn baseline
Tests: speed, convergence, accuracy (AMI, ARI, SSE)
"""
import torch
import numpy as np
import time
import json
from datetime import datetime
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans as SklearnKMeans
from mp_kmeans import KMeansPlusPlus, make_blobs_gpu
import argparse
from typing import Dict, List, Optional

import torch
import numpy as np
import random

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, inertia: float) -> Dict[str, float]:
    """
    Compute clustering metrics
    
    Returns:
        dict with AMI, ARI, and SSE
    """
    y_true_cpu = y_true.cpu().numpy()
    y_pred_cpu = y_pred.cpu().numpy()
    
    ami = adjusted_mutual_info_score(y_true_cpu, y_pred_cpu)
    ari = adjusted_rand_score(y_true_cpu, y_pred_cpu)
    
    return {
        'AMI': ami,
        'ARI': ari,
        'SSE': inertia
    }


def benchmark_sklearn_baseline(
    X: torch.Tensor,
    y_true: torch.Tensor,
    n_clusters: int,
    n_runs: int = 3,
    verbose: bool = False
) -> Dict:
    """
    Benchmark sklearn KMeans as baseline
    
    Returns:
        Dictionary with averaged results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing sklearn KMeans (CPU baseline)")
        print(f"{'='*80}")
    
    X_cpu = X.cpu().numpy()
    
    results = {
        'kernel': 'sklearn',
        'kappa': None,
        'times': [],
        'iterations': [],
        'AMI': [],
        'ARI': [],
        'SSE': [],
        'name': 'sklearn KMeans (CPU)'
    }
    
    for run in range(n_runs):
        if verbose:
            print(f"\nRun {run + 1}/{n_runs}")
        
        kmeans = SklearnKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            max_iter=300,
            tol=1e-8,
            random_state=42 + run,
            n_init=1,  # Single initialization for fair comparison
            verbose=0
        )
        
        # Fit
        t_start = time.time()
        kmeans.fit(X_cpu)
        t_end = time.time()
        
        # Compute metrics
        y_pred = torch.from_numpy(kmeans.labels_)
        metrics = compute_metrics(y_true, y_pred, kmeans.inertia_)
        
        # Record (skip first run for timing, but keep for metrics)
        if run > 0:  # Skip warmup for timing
            results['times'].append(t_end - t_start)
        
        results['iterations'].append(kmeans.n_iter_)
        results['AMI'].append(metrics['AMI'])
        results['ARI'].append(metrics['ARI'])
        results['SSE'].append(metrics['SSE'])
        
        if verbose:
            print(f"  Time: {t_end - t_start:.4f}s")
            print(f"  Iterations: {kmeans.n_iter_}")
            print(f"  AMI: {metrics['AMI']:.4f}, ARI: {metrics['ARI']:.4f}, SSE: {metrics['SSE']:.2f}")
    
    # Compute averages (excluding warmup for time)
    results['avg_time'] = np.mean(results['times'])
    results['std_time'] = np.std(results['times'])
    results['avg_iterations'] = np.mean(results['iterations'])
    results['std_iterations'] = np.std(results['iterations'])
    results['avg_AMI'] = np.mean(results['AMI'])
    results['avg_ARI'] = np.mean(results['ARI'])
    results['avg_SSE'] = np.mean(results['SSE'])
    
    return results


def benchmark_kernel(
    X: torch.Tensor,
    y_true: torch.Tensor,
    n_clusters: int,
    kernel: str,
    kappa: float = 5.0,
    n_runs: int = 3,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Benchmark a single kernel with multiple runs
    
    Returns:
        Dictionary with averaged results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing kernel: {kernel} (kappa={kappa})")
        print(f"{'='*80}")
    
    results = {
        'kernel': kernel,
        'kappa': kappa,
        'times': [],
        'iterations': [],
        'AMI': [],
        'ARI': [],
        'SSE': [],
        'init_time': [],
        'iter_times': [],
    }
    
    for run in range(n_runs):
        set_all_seeds(42 + run)
        if verbose:
            print(f"\nRun {run + 1}/{n_runs}")
        
        try:
            kmeans = KMeansPlusPlus(n_clusters=n_clusters, kernel=kernel, kappa=kappa, max_iter=300, tol=1e-8, normalize='standard',  random_state=42 + run, verbose=verbose
            )
            
            # Fit
            t_start = time.time()
            kmeans.fit(X)
            t_end = time.time()
            
            # Compute metrics
            metrics = compute_metrics(y_true, kmeans.labels_, kmeans.inertia_)
            
            # Record (skip first run for timing, but keep for metrics)
            if run > 0:  # Skip warmup for timing
                results['times'].append(t_end - t_start)
                results['init_time'].append(kmeans.timing_['init'])
                results['iter_times'].append(kmeans.timing_['iterations'])
            
            results['iterations'].append(kmeans.n_iter_)
            results['AMI'].append(metrics['AMI'])
            results['ARI'].append(metrics['ARI'])
            results['SSE'].append(metrics['SSE'])
            
            if verbose:
                print(f"  Time: {t_end - t_start:.4f}s")
                print(f"  Iterations: {kmeans.n_iter_}")
                print(f"  AMI: {metrics['AMI']:.4f}, ARI: {metrics['ARI']:.4f}, SSE: {metrics['SSE']:.2f}")
        
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            return None
    
    # Compute averages (excluding warmup for time)
    results['avg_time'] = np.mean(results['times'])
    results['std_time'] = np.std(results['times'])
    results['avg_iterations'] = np.mean(results['iterations'])
    results['std_iterations'] = np.std(results['iterations'])
    results['avg_AMI'] = np.mean(results['AMI'])
    results['avg_ARI'] = np.mean(results['ARI'])
    results['avg_SSE'] = np.mean(results['SSE'])
    results['avg_init_time'] = np.mean(results['init_time'])
    
    # Average iteration time
    all_iter_times = []
    for iter_times in results['iter_times']:
        all_iter_times.extend(iter_times)
    results['avg_iter_time'] = np.mean(all_iter_times) if all_iter_times else 0
    
    return results


def save_results_to_json(all_results: List[Dict], config: Dict, output_file: str = None):
    """
    Save benchmark results to JSON file
    
    Args:
        all_results: List of result dictionaries
        config: Benchmark configuration parameters
        output_file: Optional custom output filename
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.json"
    
    # Prepare data for JSON serialization
    json_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'pytorch_version': torch.__version__,
        },
        'config': config,
        'results': []
    }
    
    # Process each result
    for result in all_results:
        # Convert numpy types to Python native types for JSON serialization
        json_result = {}
        for key, value in result.items():
            if isinstance(value, (np.integer, np.floating)):
                json_result[key] = float(value)
            elif isinstance(value, list):
                json_result[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
            else:
                json_result[key] = value
        
        # Add speedup relative to sklearn
        baseline_time = next((r['avg_time'] for r in all_results if r['kernel'] == 'sklearn'), None)
        if baseline_time:
            json_result['speedup_vs_sklearn'] = baseline_time / json_result['avg_time']
        
        json_data['results'].append(json_result)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    return output_file


def run_comprehensive_benchmark(
    n_samples: int = 100000,
    n_features: int = 128,
    n_clusters: int = 50,
    cluster_std: float = 1.0,
    n_runs: int = 3,
    kappa_values: List[float] = [5.0],
    verbose: bool = False,
    save_json: bool = True,
    output_file: str = None
):
    """
    Run comprehensive benchmark across all kernels
    """
    print("\n" + "="*100)
    print(f"🚀 KMEANS++ PRECISION MODE BENCHMARK")
    print("="*100)
    print(f"\nDataset:")
    print(f"  Samples:   {n_samples:,}")
    print(f"  Features:  {n_features}")
    print(f"  Clusters:  {n_clusters}")
    print(f"  Std:       {cluster_std}")
    print(f"  Runs:      {n_runs} (first run is warmup)")
    print()
    
    # Store config for JSON
    config = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_clusters': n_clusters,
        'cluster_std': cluster_std,
        'n_runs': n_runs,
        'kappa_values': kappa_values,
    }
    
    # Generate data
    print("Generating synthetic data...")
    X, y_true = make_blobs_gpu(
        n_samples=n_samples,
        n_features=n_features,
        n_centers=n_clusters,
        cluster_std=cluster_std,
        random_state=42
    )
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  GPU memory: {X.element_size() * X.numel() / 1024**2:.1f} MB\n")

    # Define kernels to test
    kernels_uniform = [
        ('FP16 (uniform, PyTorch)', 'fp16_uniform', 5.0),
        ('BF16 (uniform, PyTorch)', 'bf16_uniform', 5.0),
        ('FP32 (uniform, PyTorch)', 'fp32', 5.0),
        ('FP64 (uniform, PyTorch)', 'fp64', 100.0),
    ]
    
    
    kernels_mixed_mix = [
        ('FP16+FP32 (mixed)', 'fp16_fp32', kappa_values),
        ('BF16+FP32 (mixed)', 'bf16_fp32', kappa_values),
        ('FP32+FP32 (mixed)', 'fp32_fp32', kappa_values),
        ('FP16+FP64 (mixed)', 'fp16_fp64', kappa_values),
        ('BF16+FP64 (mixed)', 'bf16_fp64', kappa_values),
        # ('TF32+FP64 (mixed)', 'tf32_fp64', kappa_values),
        ('FP32+FP64 (mixed)', 'fp32_fp64', kappa_values),
    ]
    

    all_results = []
    baseline_time = 0.0

    # Test uniform precision
    print("\n" + "="*100)
    print("CATEGORY 1: UNIFORM PRECISION (PyTorch native, No fallback)")
    print("="*100)
    
    for name, kernel, kappa in kernels_uniform:
        result = benchmark_kernel(X, y_true, n_clusters, kernel, kappa, n_runs, verbose)
        if result:
            result['name'] = name
            all_results.append(result)
            print(f"{name:<30} | Time: {result['avg_time']:.3f}s | Iter: {result['avg_iterations']:.1f} | AMI: {result['avg_AMI']:.4f}")
            if name == 'FP64 (uniform, PyTorch)':
                baseline_time = result['avg_time'] 
        
        else:
            print(f" {name:<30} | FAILED")

    print(f"\n{'='*100}")
    print("CATEGORY 2: MIXED PRECISION (Low-precision primary + FP64 fallback)")
    print("="*100)
    
    for name, kernel, kappas in kernels_mixed_mix:
        for kappa in kappas:
            result = benchmark_kernel(X, y_true, n_clusters, kernel, kappa, n_runs, verbose)
            if result:
                full_name = f"{name} (κ={kappa})"
                result['name'] = full_name
                all_results.append(result)
                print(f"{full_name:<35} | Time: {result['avg_time']:.3f}s | Iter: {result['avg_iterations']:.1f} | AMI: {result['avg_AMI']:.4f}")
            else:
                print(f" {full_name:<35} | FAILED")
    
    print(f"\n{'='*130}")
    print(f"{'COMPREHENSIVE SUMMARY':<130}")
    print(f"{'='*130}")
    print(f"{'Method':<35} {'Avg Time':>12} {'Speedup':>10} {'Iterations':>12} {'AMI':>10} {'ARI':>10} {'SSE':>12}")
    print(f"{'-'*130}")
    
    # Sort by time
    all_results_sorted = sorted(all_results, key=lambda x: x['avg_time'])
    
    
    for result in all_results_sorted:
        speedup = baseline_time / result['avg_time']
        print(f"{result['name']:<35} {result['avg_time']:>10.3f}s {speedup:>9.2f}x "
              f"{result['avg_iterations']:>11.1f} {result['avg_AMI']:>10.4f} "
              f"{result['avg_ARI']:>10.4f} {result['avg_SSE']:>12.2f}")
    
    print(f"{'='*130}\n")
    
    # Analysis
    print("🔍 KEY FINDINGS:\n")
    
    # Fastest
    fastest = min(all_results, key=lambda x: x['avg_time'])
    speedup_vs_sklearn = baseline_time / fastest['avg_time']
    print(f"   🏆 Fastest:           {fastest['name']:<35} ({fastest['avg_time']:.3f}s, {speedup_vs_sklearn:.1f}x faster than sklearn)")
    
    # Most accurate (highest AMI)
    most_accurate = max(all_results, key=lambda x: x['avg_AMI'])
    print(f"   🎯 Most Accurate:     {most_accurate['name']:<35} (AMI: {most_accurate['avg_AMI']:.4f})")
    
    # Best balance (fast + accurate, AMI > 0.95, lowest time)
    balanced = [r for r in all_results if r['avg_AMI'] > 0.95]
    if balanced:
        best_balanced = min(balanced, key=lambda x: x['avg_time'])
        speedup_balanced = baseline_time / best_balanced['avg_time']
        print(f"   ⚖️  Best Balance:      {best_balanced['name']:<35} ({best_balanced['avg_time']:.3f}s, AMI: {best_balanced['avg_AMI']:.4f}, {speedup_balanced:.1f}x vs sklearn)")
    
    # Convergence comparison
    fastest_converge = min(all_results, key=lambda x: x['avg_iterations'])
    slowest_converge = max(all_results, key=lambda x: x['avg_iterations'])
    print(f"\n   Convergence:")
    print(f"      Fastest: {fastest_converge['name']:<35} ({fastest_converge['avg_iterations']:.1f} iterations)")
    print(f"      Slowest: {slowest_converge['name']:<35} ({slowest_converge['avg_iterations']:.1f} iterations)")
    
    # Accuracy comparison
    print(f"\n   🎯 Accuracy (AMI):")
    uniform_results = [r for r in all_results if 'uniform' in r['name']]
    mixed_results = [r for r in all_results if 'mixed' in r['name']]
    
    if uniform_results:
        avg_ami_uniform = np.mean([r['avg_AMI'] for r in uniform_results])
        print(f"      Uniform modes:     {avg_ami_uniform:.4f} (avg)")
    
    if mixed_results:
        avg_ami_mixed = np.mean([r['avg_AMI'] for r in mixed_results])
        print(f"      Mixed modes:       {avg_ami_mixed:.4f} (avg)")
        if uniform_results:
            improvement = (avg_ami_mixed - avg_ami_uniform) / avg_ami_uniform * 100
            print(f"      Improvement:       {improvement:+.2f}%")
    
    print()
    
    # Kappa analysis (if multiple kappas tested)
    print("   🎚️  Kappa Parameter Analysis:")
    kappa_groups = {}
    for r in all_results:
        if 'mixed' in r['name'] or 'advanced' in r['name']:
            base_name = r['name'].split('(κ=')[0].strip()
            if base_name not in kappa_groups:
                kappa_groups[base_name] = []
            kappa_groups[base_name].append(r)
    
    for base_name, results_group in kappa_groups.items():
        if len(results_group) > 1:
            print(f"\n      {base_name}:")
            for r in sorted(results_group, key=lambda x: x['kappa']):
                speedup = baseline_time / r['avg_time']
                print(f"        κ={r['kappa']:<5} → Time: {r['avg_time']:.3f}s ({speedup:.1f}x), AMI: {r['avg_AMI']:.4f}, Iter: {r['avg_iterations']:.1f}")
    
    print("\n" + "="*130)
    print("💡 RECOMMENDATIONS")
    print("="*130 + "\n")
    
    print("Based on the results:\n")
    
    print("🟢 For SPEED priority:")
    fastest_3 = all_results_sorted[:3]
    for i, r in enumerate(fastest_3, 1):
        speedup = baseline_time / r['avg_time']
        print(f"   {i}. {r['name']:<40} ({r['avg_time']:.3f}s, {speedup:.1f}x vs sklearn, AMI: {r['avg_AMI']:.4f})")
    
    print("\n🟡 For ACCURACY priority:")
    accurate_3 = sorted(all_results, key=lambda x: x['avg_AMI'], reverse=True)[:3]
    for i, r in enumerate(accurate_3, 1):
        speedup = baseline_time / r['avg_time']
        print(f"   {i}. {r['name']:<40} (AMI: {r['avg_AMI']:.4f}, {r['avg_time']:.3f}s, {speedup:.1f}x)")
    
    print("\n🟢 For BALANCED performance:")
    if balanced:
        balanced_3 = sorted(balanced, key=lambda x: x['avg_time'])[:3]
        for i, r in enumerate(balanced_3, 1):
            speedup = baseline_time / r['avg_time']
            print(f"   {i}. {r['name']:<40} ({r['avg_time']:.3f}s, {speedup:.1f}x, AMI: {r['avg_AMI']:.4f})")
    
    print("\n" + "="*130 + "\n")
    
    # Save results to JSON
    if save_json:
        save_results_to_json(all_results, config, output_file)
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='KMeans++ multi-precision benchmark with sklearn baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python test_kmeans_precision.py                           # Standard test
                python test_kmeans_precision.py --small                   # Quick test
                python test_kmeans_precision.py --large                   # Large scale
                python test_kmeans_precision.py --kappa 3.0 5.0 10.0     # Test multiple kappas
                python test_kmeans_precision.py -v                        # Verbose output
                python test_kmeans_precision.py --output my_results.json  # Custom output file
                python test_kmeans_precision.py --no-save                 # Don't save JSON
               """)
    
    parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples')
    parser.add_argument('--n_features', type=int, default=128, help='Number of features')
    parser.add_argument('--n_clusters', type=int, default=100, help='Number of clusters')
    parser.add_argument('--cluster_std', type=float, default=1.0, help='Cluster standard deviation')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs (first is warmup)')
    parser.add_argument('--kappa', nargs='+', type=float, default=[10.0], help='Kappa values to test')
    parser.add_argument('--small', action='store_true', help='Small test (10k samples, 10 clusters)')
    parser.add_argument('--large', action='store_true', help='Large test (500k samples, 100 clusters)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', type=str, default=None, help='Custom output JSON filename')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to JSON')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print(" CUDA not available!")
        exit(1)
    
    # Print system info
    print("\n" + "="*100)
    print("🖥️  System Information")
    print("="*100)
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print("="*100)
    
    # Adjust parameters for small/large tests
    if args.small:
        args.n_samples = 10000
        args.n_features = 64
        args.n_clusters = 10
        print("\n🔸 Running SMALL test")
    elif args.large:
        args.n_samples = 500000
        args.n_features = 256
        args.n_clusters = 100
        print("\n🔶 Running LARGE test")
    
    # Run benchmark
    results = run_comprehensive_benchmark(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        cluster_std=args.cluster_std,
        n_runs=args.n_runs,
        kappa_values=args.kappa,
        verbose=args.verbose,
        save_json=not args.no_save,
        output_file=args.output
    )