#!/usr/bin/env python3
"""
Performance verification test for HyperTokenizer16k smart optimizations
"""
import time
import tiktoken
from new_trainer import HyperTokenizer16k

def quick_benchmark(tokenizer, text, name, iterations=100):
    # Warm up
    for _ in range(5):
        tokens = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else tokenizer.encode(text)
    
    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        tokens = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else tokenizer.encode(text)
    end = time.perf_counter()
    
    avg_time = (end - start) / iterations * 1000  # ms
    return len(tokens), avg_time

# Load tokenizers
print("Loading tokenizers...")
hyper_tok = HyperTokenizer16k.load("ultra_hyper_tokenizer_16k.pkl")
tiktoken_enc = tiktoken.get_encoding("o200k_base")

# Test cases
test_cases = [
    "Hello world! This is a test of the HyperTokenizer system.",
    "The quick brown fox jumps over the lazy dog 123 times.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "Natural language processing (NLP) and artificial intelligence (AI) are revolutionizing technology.",
    "🚀 Check out https://github.com/user/repo for more info! @username #AI"
]

print("\n" + "="*70)
print("PERFORMANCE VERIFICATION RESULTS")
print("="*70)

hyper_times = []
tiktoken_times = []
compression_ratios = []

for i, text in enumerate(test_cases, 1):
    print(f"\nTest {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    # Benchmark HyperTokenizer
    hyper_tokens, hyper_time = quick_benchmark(hyper_tok, text, "HyperTokenizer")
    hyper_times.append(hyper_time)
    
    # Benchmark tiktoken
    tiktoken_tokens, tiktoken_time = quick_benchmark(tiktoken_enc, text, "tiktoken")
    tiktoken_times.append(tiktoken_time)
    
    # Calculate compression ratio
    compression_ratio = hyper_tokens / tiktoken_tokens if tiktoken_tokens > 0 else 1.0
    compression_ratios.append(compression_ratio)
    
    print(f"  HyperTokenizer: {hyper_tokens:3d} tokens, {hyper_time:6.2f} ms")
    print(f"  tiktoken:       {tiktoken_tokens:3d} tokens, {tiktoken_time:6.2f} ms")
    print(f"  Compression:    {compression_ratio:.3f} (lower is better)")
    print(f"  Speed ratio:    {hyper_time/tiktoken_time:.2f}x (lower is faster)")

# Overall results
print(f"\n{'='*70}")
print("OVERALL RESULTS")
print(f"{'='*70}")

avg_hyper_time = sum(hyper_times) / len(hyper_times)
avg_tiktoken_time = sum(tiktoken_times) / len(tiktoken_times)
avg_compression = sum(compression_ratios) / len(compression_ratios)

print(f"Average HyperTokenizer time: {avg_hyper_time:.2f} ms")
print(f"Average tiktoken time:       {avg_tiktoken_time:.2f} ms")
print(f"Average compression ratio:   {avg_compression:.3f}")

speed_ratio = avg_hyper_time / avg_tiktoken_time
if speed_ratio < 1.0:
    print(f"\n🚀 HyperTokenizer is {1/speed_ratio:.2f}x FASTER than tiktoken!")
else:
    print(f"\n⚠️  HyperTokenizer is {speed_ratio:.2f}x slower than tiktoken")

if avg_compression < 1.0:
    improvement = (1.0 - avg_compression) * 100
    print(f"🎯 HyperTokenizer uses {improvement:.1f}% fewer tokens than tiktoken!")
else:
    overhead = (avg_compression - 1.0) * 100
    print(f"📈 HyperTokenizer uses {overhead:.1f}% more tokens than tiktoken")

print(f"\n{'='*70}")
