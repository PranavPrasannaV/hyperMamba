import os
import time
import tiktoken
import statistics
import random
import string
from hypertokenizer_16k import HyperTokenizer16k


PKL_PATH = "ultra_hyper_tokenizer_16k.pkl"

def benchmark_tokenizer(tok, text):
    t0 = time.time()
    ids = tok.encode(text, use_lattice=False)
    t1 = time.time()
    decoded = tok.decode(ids)
    return {
        "tokens": len(ids),
        "encode_time_ms": (t1 - t0) * 1000,
        "roundtrip_ok": decoded == text
    }

def benchmark_tiktoken(encoding, text):
    t0 = time.time()
    ids = encoding.encode(text)
    t1 = time.time()
    # tiktoken decode takes IDs back to text
    decoded = encoding.decode(ids)
    return {
        "tokens": len(ids),
        "encode_time_ms": (t1 - t0) * 1000,
        "roundtrip_ok": decoded == text
    }

def generate_test_texts():
    """Generate comprehensive test cases for tokenizer benchmarking"""
    
    # Original sample texts (keeping existing ones)
    original_samples = [
        "Hello world! This is a test of the HyperTokenizer system.",
        "The quick brown fox jumps over the lazy dog 123 times.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Natural language processing (NLP) and artificial intelligence (AI) are revolutionizing technology.",
        "🚀 Check out https://github.com/user/repo for more info! @username #AI"
    ]
    
    # Edge cases
    edge_cases = [
        "",  # Empty string
        " ",  # Single space
        "\n",  # Single newline
        "\t",  # Single tab
        "a",  # Single character
        "123",  # Numbers only
        "!@#$%^&*()",  # Special characters only
        "   multiple   spaces   ",  # Multiple spaces
        "\n\n\nmultiple\n\nnewlines\n\n",  # Multiple newlines
        "CamelCaseWords",  # CamelCase
        "snake_case_words",  # Snake case
        "kebab-case-words",  # Kebab case
        "ALLCAPSTEXT",  # All caps
        "MiXeD cAsE tExT",  # Mixed case
    ]
    
    # Unicode and multilingual text
    unicode_tests = [
        "Hello 世界! Bonjour le monde! ¡Hola mundo!",  # Mixed languages
        "🌍🌎🌏 Earth emojis with text",  # Emojis
        "Café naïve résumé",  # Accented characters
        "Здравствуй мир",  # Cyrillic
        "こんにちは世界",  # Japanese
        "مرحبا بالعالم",  # Arabic
        "שלום עולם",  # Hebrew
        "नमस्ते दुनिया",  # Hindi
        "𝕳𝖊𝖑𝖑𝖔 𝖂𝖔𝖗𝖑𝖉",  # Mathematical symbols
        "α β γ δ ε ζ η θ",  # Greek letters
    ]
    
    # Code samples in different languages
    code_samples = [
        # Python
        '''
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
        '''.strip(),
        
        # JavaScript
        '''
const fibonacci = (n) => {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
};
console.log(fibonacci(10));
        '''.strip(),
        
        # HTML/CSS
        '''
<!DOCTYPE html>
<html>
<head>
    <style>
        .container { display: flex; justify-content: center; }
        .item { background: #f0f0f0; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="item">Hello World</div>
    </div>
</body>
</html>
        '''.strip(),
        
        # SQL
        '''
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2023-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5
ORDER BY order_count DESC;
        '''.strip(),
        
        # JSON
        '''
{
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "hobbies": ["reading", "swimming", "coding"],
    "address": {
        "street": "123 Main St",
        "zipcode": "10001"
    }
}
        '''.strip(),
    ]
    
    # Long text samples
    long_texts = [
        # Repeated pattern
        "The quick brown fox jumps over the lazy dog. " * 100,
        
        # Lorem ipsum
        '''Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.''' * 10,
        
        # Technical documentation style
        '''The HyperTokenizer16k is an advanced tokenization system designed for high-performance natural language processing tasks. It implements a sophisticated byte-pair encoding (BPE) algorithm with lattice-based optimization for improved compression ratios and processing speed. The tokenizer supports Unicode normalization, handles out-of-vocabulary tokens gracefully, and provides efficient encoding and decoding operations suitable for large-scale language model training and inference.''' * 20,
    ]
    
    # Random text generation
    random_texts = []
    for length in [10, 50, 100, 500, 1000]:
        # Random ASCII text
        random_ascii = ''.join(random.choices(string.ascii_letters + string.digits + ' .,!?', k=length))
        random_texts.append(random_ascii)
        
        # Random words
        words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'hello', 'world', 'python', 'code', 'test', 'sample']
        random_words = ' '.join(random.choices(words, k=length//5))
        random_texts.append(random_words)
    
    # Whitespace variations
    whitespace_tests = [
        "word1\tword2\tword3",  # Tabs
        "word1\nword2\nword3",  # Newlines
        "word1\r\nword2\r\nword3",  # Windows line endings
        "word1 \n \t word2",  # Mixed whitespace
        "   leading spaces",
        "trailing spaces   ",
        "  both  sides  ",
    ]
    
    # URL and email patterns
    web_patterns = [
        "Visit https://www.example.com for more information",
        "Contact us at support@example.com or admin@test.org",
        "Check out http://github.com/user/repo and https://stackoverflow.com/questions/123456",
        "ftp://files.example.com/download/file.zip",
        "mailto:user@domain.com?subject=Hello&body=Test message",
    ]
    
    # Numeric patterns
    numeric_tests = [
        "1234567890",
        "3.14159265359",
        "1,000,000.50",
        "$1,234.56 €789.01 ¥12345",
        "Phone: +1-555-123-4567",
        "Date: 2024-01-15 Time: 14:30:25",
        "Version 1.2.3-beta.4+build.567",
        "IPv4: 192.168.1.1 IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334",
    ]
    
    return {
        'original': original_samples,
        'edge_cases': edge_cases,
        'unicode': unicode_tests,
        'code': code_samples,
        'long_texts': long_texts,
        'random': random_texts,
        'whitespace': whitespace_tests,
        'web_patterns': web_patterns,
        'numeric': numeric_tests,
    }

def run_comprehensive_benchmark(hyper_tok, tiktoken_enc, test_categories):
    """Run comprehensive benchmarks across all test categories"""
    
    results = {}
    total_tests = 0
    
    for category, texts in test_categories.items():
        print(f"\n{'='*80}")
        print(f"BENCHMARK CATEGORY: {category.upper()}")
        print(f"{'='*80}")
        
        category_results = {
            'hyper_times': [],
            'tiktoken_times': [],
            'hyper_tokens': [],
            'tiktoken_tokens': [],
            'hyper_failures': 0,
            'tiktoken_failures': 0,
            'compression_ratios': [],
        }
        
        for i, text in enumerate(texts):
            total_tests += 1
            
            try:
                # Benchmark HyperTokenizer
                ht_res = benchmark_tokenizer(hyper_tok, text)
                category_results['hyper_times'].append(ht_res['encode_time_ms'])
                category_results['hyper_tokens'].append(ht_res['tokens'])
                if not ht_res['roundtrip_ok']:
                    category_results['hyper_failures'] += 1
                    
            except Exception as e:
                print(f"HyperTokenizer failed on text {i}: {e}")
                category_results['hyper_failures'] += 1
                category_results['hyper_times'].append(float('inf'))
                category_results['hyper_tokens'].append(0)
            
            try:
                # Benchmark tiktoken
                tk_res = benchmark_tiktoken(tiktoken_enc, text)
                category_results['tiktoken_times'].append(tk_res['encode_time_ms'])
                category_results['tiktoken_tokens'].append(tk_res['tokens'])
                if not tk_res['roundtrip_ok']:
                    category_results['tiktoken_failures'] += 1
                    
            except Exception as e:
                print(f"Tiktoken failed on text {i}: {e}")
                category_results['tiktoken_failures'] += 1
                category_results['tiktoken_times'].append(float('inf'))
                category_results['tiktoken_tokens'].append(0)
            
            # Calculate compression ratio (HyperTokenizer tokens / tiktoken tokens)
            if category_results['tiktoken_tokens'][-1] > 0:
                ratio = category_results['hyper_tokens'][-1] / category_results['tiktoken_tokens'][-1]
                category_results['compression_ratios'].append(ratio)
            
            # Print individual results for shorter texts
            if len(text) <= 200:
                print(f"\nText {i+1}: {text[:100]}{'...' if len(text) > 100 else ''}")
                try:
                    print(f"  HyperTokenizer → Tokens: {category_results['hyper_tokens'][-1]}, Time: {category_results['hyper_times'][-1]:.2f} ms")
                    print(f"  Tiktoken      → Tokens: {category_results['tiktoken_tokens'][-1]}, Time: {category_results['tiktoken_times'][-1]:.2f} ms")
                    if category_results['compression_ratios']:
                        print(f"  Compression Ratio: {category_results['compression_ratios'][-1]:.3f}")
                except:
                    pass
        
        # Calculate statistics for this category
        if category_results['hyper_times'] and category_results['tiktoken_times']:
            valid_hyper_times = [t for t in category_results['hyper_times'] if t != float('inf')]
            valid_tiktoken_times = [t for t in category_results['tiktoken_times'] if t != float('inf')]
            
            print(f"\n{'-'*60}")
            print(f"CATEGORY SUMMARY: {category.upper()}")
            print(f"{'-'*60}")
            print(f"Total texts tested: {len(texts)}")
            
            if valid_hyper_times:
                print(f"HyperTokenizer:")
                print(f"  Avg time: {statistics.mean(valid_hyper_times):.2f} ms")
                print(f"  Med time: {statistics.median(valid_hyper_times):.2f} ms")
                print(f"  Min time: {min(valid_hyper_times):.2f} ms")
                print(f"  Max time: {max(valid_hyper_times):.2f} ms")
                print(f"  Failures: {category_results['hyper_failures']}")
            
            if valid_tiktoken_times:
                print(f"Tiktoken:")
                print(f"  Avg time: {statistics.mean(valid_tiktoken_times):.2f} ms")
                print(f"  Med time: {statistics.median(valid_tiktoken_times):.2f} ms")
                print(f"  Min time: {min(valid_tiktoken_times):.2f} ms")
                print(f"  Max time: {max(valid_tiktoken_times):.2f} ms")
                print(f"  Failures: {category_results['tiktoken_failures']}")
            
            if category_results['compression_ratios']:
                print(f"Compression (HyperTokenizer/Tiktoken ratio):")
                print(f"  Avg ratio: {statistics.mean(category_results['compression_ratios']):.3f}")
                print(f"  Med ratio: {statistics.median(category_results['compression_ratios']):.3f}")
                print(f"  Best ratio: {min(category_results['compression_ratios']):.3f}")
                print(f"  Worst ratio: {max(category_results['compression_ratios']):.3f}")
        
        results[category] = category_results
    
    return results, total_tests

if __name__ == "__main__":
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"{PKL_PATH} not found")

    print(f"[INFO] Loading HyperTokenizer16k from {PKL_PATH}…")
    hyper_tok = HyperTokenizer16k.load(PKL_PATH)

    print("[INFO] Loading tiktoken (o200k_base)…")
    tiktok_enc = tiktoken.get_encoding("o200k_base")

    # Run original benchmarks first (unchanged)
    print("\n" + "="*80)
    print("ORIGINAL BENCHMARK TESTS")
    print("="*80)
    
    sample_texts = [
        "Hello world! This is a test of the HyperTokenizer system.",
        "The quick brown fox jumps over the lazy dog 123 times.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Natural language processing (NLP) and artificial intelligence (AI) are revolutionizing technology.",
        "🚀 Check out https://github.com/user/repo for more info! @username #AI"
    ]

    for txt in sample_texts:
        ht_res = benchmark_tokenizer(hyper_tok, txt)
        tk_res = benchmark_tiktoken(tiktok_enc, txt)

        print("\nTEXT:", txt)
        print(f"HyperTokenizer16k → Tokens: {ht_res['tokens']}, Encode: {ht_res['encode_time_ms']:.2f} ms, Match: {ht_res['roundtrip_ok']}")
        print(f"Tiktoken (o200k_base) → Tokens: {tk_res['tokens']}, Encode: {tk_res['encode_time_ms']:.2f} ms, Match: {tk_res['roundtrip_ok']}")
        print("-" * 70)

    # Run comprehensive benchmarks
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("="*80)
    
    # Generate all test cases
    test_categories = generate_test_texts()
    
    # Run comprehensive benchmarks
    results, total_tests = run_comprehensive_benchmark(hyper_tok, tiktok_enc, test_categories)
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests run: {total_tests}")
    
    # Aggregate statistics across all categories
    all_hyper_times = []
    all_tiktoken_times = []
    all_compression_ratios = []
    total_hyper_failures = 0
    total_tiktoken_failures = 0
    
    for category, data in results.items():
        all_hyper_times.extend([t for t in data['hyper_times'] if t != float('inf')])
        all_tiktoken_times.extend([t for t in data['tiktoken_times'] if t != float('inf')])
        all_compression_ratios.extend(data['compression_ratios'])
        total_hyper_failures += data['hyper_failures']
        total_tiktoken_failures += data['tiktoken_failures']
    
    if all_hyper_times and all_tiktoken_times:
        print(f"\nOverall Performance:")
        print(f"HyperTokenizer16k:")
        print(f"  Average encode time: {statistics.mean(all_hyper_times):.2f} ms")
        print(f"  Median encode time: {statistics.median(all_hyper_times):.2f} ms")
        print(f"  Total failures: {total_hyper_failures}")
        print(f"  Success rate: {((total_tests - total_hyper_failures) / total_tests * 100):.1f}%")
        
        print(f"Tiktoken (o200k_base):")
        print(f"  Average encode time: {statistics.mean(all_tiktoken_times):.2f} ms")
        print(f"  Median encode time: {statistics.median(all_tiktoken_times):.2f} ms")
        print(f"  Total failures: {total_tiktoken_failures}")
        print(f"  Success rate: {((total_tests - total_tiktoken_failures) / total_tests * 100):.1f}%")
        
        # Speed comparison
        hyper_avg = statistics.mean(all_hyper_times)
        tiktoken_avg = statistics.mean(all_tiktoken_times)
        if hyper_avg < tiktoken_avg:
            speedup = tiktoken_avg / hyper_avg
            print(f"\n🚀 HyperTokenizer16k is {speedup:.2f}x FASTER than tiktoken on average!")
        else:
            slowdown = hyper_avg / tiktoken_avg
            print(f"\n⚠️  HyperTokenizer16k is {slowdown:.2f}x slower than tiktoken on average")
        
        if all_compression_ratios:
            avg_compression = statistics.mean(all_compression_ratios)
            print(f"\nCompression Analysis:")
            print(f"  Average compression ratio: {avg_compression:.3f}")
            if avg_compression < 1.0:
                improvement = (1.0 - avg_compression) * 100
                print(f"  🎯 HyperTokenizer16k uses {improvement:.1f}% fewer tokens on average!")
            else:
                overhead = (avg_compression - 1.0) * 100
                print(f"  📈 HyperTokenizer16k uses {overhead:.1f}% more tokens on average")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
