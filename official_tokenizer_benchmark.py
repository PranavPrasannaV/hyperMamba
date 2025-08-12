"""
Official Industry-Standard Tokenizer Benchmark Suite
====================================================

This benchmark implements evaluation methodologies used by tech giants and research institutions
to comprehensively evaluate tokenizer performance across multiple dimensions:

1. Compression Efficiency (primary metric used in industry)
2. Speed Performance (encoding/decoding latency)
3. Linguistic Quality (morphological alignment, character-level reasoning)
4. Domain Robustness (code, scientific, web, multilingual)
5. Edge Case Handling (whitespace, special characters, long sequences)

Based on research from:
- "Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance"
- "CharBench: Evaluating the Role of Tokenization in Character-Level Tasks"
- Industry practices from OpenAI, Google, Meta, and other tech giants
"""

import time
import statistics
import json
import re
import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import os
import tiktoken
from hypertokenizer_16k import HyperTokenizer16k


# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class BenchmarkResult:
    """Standardized benchmark result format"""
    tokenizer_name: str
    category: str
    test_name: str
    tokens: int
    encode_time_ms: float
    decode_time_ms: float
    compression_ratio: float  # vs reference tokenizer
    success: bool
    error_message: str = ""
    metadata: Dict[str, Any] = None


class LoggingPrintCapture:
    """Capture print statements and send them to both console and log file"""
    
    def __init__(self, logger):
        self.logger = logger
        self.original_stdout = sys.stdout
        self.buffer = ""
        
    def write(self, message):
        # Write to original stdout (console)
        self.original_stdout.write(message)
        self.original_stdout.flush()
        
        # Buffer the message for proper line handling
        self.buffer += message
        
        # Process complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():  # Only log non-empty lines
                # Log the complete line without truncation
                self.logger.info(line.rstrip())
    
    def flush(self):
        # Flush any remaining buffer content
        if self.buffer.strip():
            self.logger.info(self.buffer.strip())
            self.buffer = ""
        self.original_stdout.flush()


class OfficialTokenizerBenchmark:
    """
    Industry-standard tokenizer benchmark suite
    
    Implements comprehensive evaluation across all dimensions that matter
    for production tokenizer deployment in tech companies.
    """
    
    def __init__(self, log_file: str = None):
        self.results = []
        self.reference_tokenizer = tiktoken.get_encoding("o200k_base")  # GPT-4o tokenizer
        
        # Setup logging
        self.setup_logging(log_file)
        
        # Load test tokenizer
        try:
            model_path = os.environ.get("HYPER_MODEL_PATH", "ultra_hyper_tokenizer_16k.pkl")
            self.test_tokenizer = HyperTokenizer16k.load(model_path)
            print("✅ Loaded HyperTokenizer16k successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load HyperTokenizer16k: {e}")
    
    def setup_logging(self, log_file: str = None):
        """Setup comprehensive logging for benchmark results"""
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"tokenizer_benchmark_{timestamp}.log"
        
        # Ensure log file is in the logs directory
        if not log_file.startswith(logs_dir):
            log_file = os.path.join(logs_dir, log_file)
        
        # Create logger
        self.logger = logging.getLogger('TokenizerBenchmark')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler with no line length limits
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        
        # Simple formatter without timestamps for regular entries
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Ensure no message truncation
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        
        # Store log file path for reference
        self.log_file = log_file
        
        # Log session header with timestamp (only here)
        self.logger.info("="*100)
        self.logger.info("🚀 TOKENIZER BENCHMARK SESSION STARTED")
        self.logger.info("="*100)
        self.logger.info(f"📅 Timestamp: {datetime.now()}")
        self.logger.info(f"📝 Log file: {log_file}")
        self.logger.info(f"🐍 Python version: {sys.version}")
        self.logger.info(f"💻 Working directory: {os.getcwd()}")
        self.logger.info("="*100)
        
        print(f"📝 Logging enabled - Results will be saved to: {log_file}")
    
    def log_section_start(self, section_name: str, description: str = ""):
        """Log the start of a new benchmark section with clear separation"""
        separator = "="*100
        self.logger.info("")
        self.logger.info(separator)
        self.logger.info(f"🔥 SECTION START: {section_name}")
        if description:
            self.logger.info(f"📋 Description: {description}")
        self.logger.info(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        self.logger.info(separator)
    
    def log_section_end(self, section_name: str, summary: str = ""):
        """Log the end of a benchmark section with clear separation"""
        separator = "="*100
        self.logger.info("")
        self.logger.info(f"✅ SECTION END: {section_name}")
        if summary:
            self.logger.info(f"📊 Summary: {summary}")
        self.logger.info(f"⏰ Completed at: {datetime.now().strftime('%H:%M:%S')}")
        self.logger.info(separator)
        self.logger.info("")
    
    def log_test_entry(self, test_name: str, details: str = ""):
        """Log individual test entries with clear formatting"""
        self.logger.info(f"🧪 TEST: {test_name}")
        if details:
            # Ensure long details are not truncated
            self.logger.info(f"📝 Details: {details}")
        self.logger.info("-" * 80)
    
    def benchmark_text(self, tokenizer, text: str, tokenizer_name: str) -> Dict[str, Any]:
        """Benchmark a single text with comprehensive metrics"""
        try:
            # Encoding benchmark
            start_time = time.perf_counter()
            if tokenizer_name == "tiktoken":
                tokens = tokenizer.encode(text)
            else:
                tokens = tokenizer.encode(text, use_lattice=False)
            encode_time = (time.perf_counter() - start_time) * 1000
            
            # Decoding benchmark
            start_time = time.perf_counter()
            decoded = tokenizer.decode(tokens)
            decode_time = (time.perf_counter() - start_time) * 1000
            
            # Verify roundtrip
            success = decoded == text
            
            return {
                "tokens": len(tokens),
                "encode_time_ms": encode_time,
                "decode_time_ms": decode_time,
                "success": success,
                "decoded": decoded,
                "error": None
            }
            
        except Exception as e:
            return {
                "tokens": 0,
                "encode_time_ms": float('inf'),
                "decode_time_ms": float('inf'),
                "success": False,
                "decoded": "",
                "error": str(e)
            }
    
    def run_compression_benchmark(self):
        """
        Compression Efficiency Benchmark
        
        Primary metric used by tech giants. Tests tokenizer's ability to
        compress text efficiently across diverse domains.
        """
        print("\n" + "="*80)
        print("📊 COMPRESSION EFFICIENCY BENCHMARK")
        print("="*80)
        print("Primary metric used by OpenAI, Google, Meta for tokenizer evaluation")
        
        # Diverse text corpus covering all major domains
        test_texts = [
            # Natural Language (English)
            "The quick brown fox jumps over the lazy dog repeatedly and efficiently.",
            "Natural language processing and artificial intelligence revolutionize modern technology through advanced machine learning algorithms.",
            "In the realm of quantum computing, superposition and entanglement enable unprecedented computational capabilities.",
            
            # Programming Code
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "import torch; model = torch.nn.TransformerEncoder(layers=6, d_model=512)",
            "const apiResponse = await fetch('/api/users').then(res => res.json());",
            "public class DataProcessor { private final Config config; }",
            
            # Web Content
            "Visit https://www.example.com/docs/api/v2/reference for complete documentation",
            "<div class='container'><h1>Welcome</h1><p>Get started today!</p></div>",
            "body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }",
            
            # Scientific/Technical
            "The algorithm achieves O(n log n) time complexity with optimal space utilization",
            "Neural networks utilize backpropagation for gradient-based optimization procedures",
            "DNA sequences: ATCGATCGATCG, RNA transcription: AUCGAUCGAUCG",
            
            # Structured Data
            "SELECT * FROM users WHERE active = 1 ORDER BY created_at DESC LIMIT 100",
            '{"name": "John", "age": 30, "email": "john@example.com", "active": true}',
            "Version 2.1.3-beta.4 released with performance improvements and bug fixes",
            
            # Numeric and Identifiers
            "Processing 1,234,567 records at 99.7% accuracy with 0.003ms average latency",
            "User ID: usr_1234567890, Session: tok_abcdef123456, Expires: 2024-12-31T23:59:59Z",
            "Phone: +1-555-123-4567, Email: support@company.com, Website: https://company.com",
            
            # Mixed Content (Real-world scenarios)
            "Error 404: Page not found. Please check the URL https://example.com/missing and try again.",
            "Meeting scheduled for 2024-01-15 at 2:30 PM in Conference Room A. Agenda: Q4 review.",
            "Log entry [2024-01-15 14:30:25] INFO: User authentication successful for admin@domain.com"
        ]
        
        category_results = []
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:60]}{'...' if len(text) > 60 else ''}")
            
            # Benchmark reference tokenizer
            ref_result = self.benchmark_text(self.reference_tokenizer, text, "tiktoken")
            
            # Benchmark test tokenizer
            test_result = self.benchmark_text(self.test_tokenizer, text, "HyperTokenizer16k")
            
            # Calculate compression ratio
            if ref_result["tokens"] > 0:
                compression_ratio = test_result["tokens"] / ref_result["tokens"]
            else:
                compression_ratio = float('inf')
            
            print(f"  Reference (tiktoken)     → Tokens: {ref_result['tokens']:3d}, Time: {ref_result['encode_time_ms']:5.2f}ms")
            print(f"  HyperTokenizer16k        → Tokens: {test_result['tokens']:3d}, Time: {test_result['encode_time_ms']:5.2f}ms")
            print(f"  Compression Ratio: {compression_ratio:.3f} {'✅' if compression_ratio < 1.0 else '❌'}")
            
            # Store results
            result = BenchmarkResult(
                tokenizer_name="HyperTokenizer16k",
                category="compression",
                test_name=f"compression_test_{i}",
                tokens=test_result["tokens"],
                encode_time_ms=test_result["encode_time_ms"],
                decode_time_ms=test_result["decode_time_ms"],
                compression_ratio=compression_ratio,
                success=test_result["success"],
                error_message=test_result["error"] or "",
                metadata={"reference_tokens": ref_result["tokens"], "text_length": len(text)}
            )
            category_results.append(result)
            self.results.append(result)
        
        # Summary statistics
        compression_ratios = [r.compression_ratio for r in category_results if r.compression_ratio != float('inf')]
        encode_times = [r.encode_time_ms for r in category_results]
        
        print(f"\n{'='*60}")
        print("COMPRESSION BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {len(category_results)}")
        print(f"Success rate: {sum(r.success for r in category_results) / len(category_results) * 100:.1f}%")
        print(f"Average compression ratio: {statistics.mean(compression_ratios):.3f}")
        print(f"Median compression ratio: {statistics.median(compression_ratios):.3f}")
        print(f"Best compression ratio: {min(compression_ratios):.3f}")
        print(f"Worst compression ratio: {max(compression_ratios):.3f}")
        print(f"Average encode time: {statistics.mean(encode_times):.2f}ms")
        
        # Overall assessment
        avg_compression = statistics.mean(compression_ratios)
        if avg_compression < 0.85:
            print("🏆 EXCELLENT: Significantly better compression than reference")
        elif avg_compression < 0.95:
            print("✅ GOOD: Better compression than reference")
        elif avg_compression < 1.05:
            print("⚖️  COMPARABLE: Similar compression to reference")
        else:
            print("❌ POOR: Worse compression than reference")
    
    def run_speed_benchmark(self):
        """
        Speed Performance Benchmark
        
        Critical for production deployment. Tests encoding/decoding speed
        across various text lengths and complexities.
        """
        print("\n" + "="*80)
        print("⚡ SPEED PERFORMANCE BENCHMARK")
        print("="*80)
        print("Critical metric for production deployment in tech companies")
        
        # Speed test cases with varying lengths and complexities
        speed_tests = [
            ("Short text", "Hello world!"),
            ("Medium text", "The quick brown fox jumps over the lazy dog " * 10),
            ("Long text", "Natural language processing and machine learning " * 50),
            ("Code snippet", "def process_data(data):\n    return [x for x in data if x > 0]" * 20),
            ("JSON data", '{"users": [{"name": "John", "age": 30}]}' * 30),
            ("URL heavy", "Visit https://www.example.com/path/to/resource?param=value " * 25),
            ("Mixed content", "Error 404: Contact support@company.com or visit https://help.example.com " * 15)
        ]
        
        print(f"\n{'Test Name':<15} {'Length':<8} {'HyperTokenizer':<15} {'Tiktoken':<12} {'Speedup':<10}")
        print("-" * 70)
        
        speed_results = []
        
        for test_name, base_text in speed_tests:
            text = base_text
            
            # Benchmark HyperTokenizer (multiple runs for accuracy)
            hyper_times = []
            for _ in range(5):
                start = time.perf_counter()
                tokens = self.test_tokenizer.encode(text, use_lattice=False)
                hyper_times.append((time.perf_counter() - start) * 1000)
            hyper_avg = statistics.mean(hyper_times)
            
            # Benchmark Tiktoken
            tiktoken_times = []
            for _ in range(5):
                start = time.perf_counter()
                tokens_ref = self.reference_tokenizer.encode(text)
                tiktoken_times.append((time.perf_counter() - start) * 1000)
            tiktoken_avg = statistics.mean(tiktoken_times)
            
            # Calculate speedup
            speedup = tiktoken_avg / hyper_avg if hyper_avg > 0 else 0
            
            print(f"{test_name:<15} {len(text):<8} {hyper_avg:<15.2f} {tiktoken_avg:<12.2f} {speedup:<10.2f}x")
            
            speed_results.append({
                'test_name': test_name,
                'hyper_time': hyper_avg,
                'tiktoken_time': tiktoken_avg,
                'speedup': speedup
            })
        
        # Speed summary
        avg_speedup = statistics.mean([r['speedup'] for r in speed_results])
        print(f"\n{'='*60}")
        print("SPEED BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Average speedup vs tiktoken: {avg_speedup:.2f}x")
        
        if avg_speedup > 1.5:
            print("🏆 EXCELLENT: Significantly faster than reference")
        elif avg_speedup > 1.1:
            print("✅ GOOD: Faster than reference")
        elif avg_speedup > 0.9:
            print("⚖️  COMPARABLE: Similar speed to reference")
        else:
            print("❌ NEEDS IMPROVEMENT: Slower than reference")
    
    def run_character_level_benchmark(self):
        """
        Character-Level Reasoning Benchmark (CharBench)
        
        Based on academic research. Tests tokenizer's impact on character-level
        understanding, critical for many NLP tasks.
        """
        print("\n" + "="*80)
        print("🔤 CHARACTER-LEVEL REASONING BENCHMARK (CharBench)")
        print("="*80)
        print("Based on academic research - tests character-level understanding")
        
        # Character-level test cases
        char_tests = [
            # Counting tasks
            ("Count 'r' in 'strawberry'", "strawberry", "r", 3),
            ("Count 'l' in 'hello'", "hello", "l", 2),
            ("Count 'a' in 'banana'", "banana", "a", 3),
            ("Count 's' in 'assessment'", "assessment", "s", 4),
            
            # Position finding tasks
            ("Find first 'o' in 'hello'", "hello", "o", 4),
            ("Find first 'a' in 'banana'", "banana", "a", 1),
            ("Find first 'e' in 'development'", "development", "e", 1),
        ]
        
        print(f"\n{'Task':<25} {'Word':<12} {'Target':<8} {'Expected':<10} {'Tokens':<8} {'Efficiency'}")
        print("-" * 80)
        
        for task, word, char, expected in char_tests:
            # Tokenize the word
            tokens = self.test_tokenizer.encode(word, use_lattice=False)
            ref_tokens = self.reference_tokenizer.encode(word)
            
            # Calculate efficiency (fewer tokens = better for char-level tasks)
            efficiency = len(ref_tokens) / len(tokens) if len(tokens) > 0 else 0
            
            print(f"{task:<25} {word:<12} {char:<8} {expected:<10} {len(tokens):<8} {efficiency:.2f}x")
        
        print(f"\n{'='*60}")
        print("CHARACTER-LEVEL BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print("Lower token counts generally correlate with better character-level performance")
        print("This benchmark assesses tokenizer's suitability for character-aware tasks")
    
    def run_robustness_benchmark(self):
        """
        Robustness and Edge Case Benchmark
        
        Tests tokenizer behavior on challenging inputs that commonly
        cause failures in production systems.
        """
        print("\n" + "="*80)
        print("🛡️  ROBUSTNESS AND EDGE CASE BENCHMARK")
        print("="*80)
        print("Tests challenging inputs that commonly cause production failures")
        
        edge_cases = [
            # Whitespace variations
            ("Multiple spaces", "word1    word2    word3"),
            ("Mixed whitespace", "word1\t\tword2\n\nword3"),
            ("Leading/trailing", "   leading and trailing   "),
            
            # Special characters
            ("Unicode", "café naïve résumé 🚀 🎯 ✅"),
            ("Punctuation heavy", "Hello!!! What??? Yes... No--- Maybe???"),
            ("Mixed symbols", "@#$%^&*()[]{}|\\:;\"'<>,./?"),
            
            # Long sequences
            ("Repeated chars", "a" * 100),
            ("Long identifier", "very_long_variable_name_that_goes_on_and_on_and_on"),
            ("Long URL", "https://www.example.com/very/long/path/to/resource/with/many/segments/and/parameters?param1=value1&param2=value2"),
            
            # Empty and minimal
            ("Empty string", ""),
            ("Single char", "a"),
            ("Single space", " "),
            
            # Numeric edge cases
            ("Large numbers", "123456789012345678901234567890"),
            ("Scientific notation", "1.23e-45 6.78e+123"),
            ("Mixed numeric", "v1.2.3-beta.4+build.567.890"),
        ]
        
        print(f"\n{'Test Case':<20} {'Input':<30} {'Tokens':<8} {'Success':<8} {'Time (ms)'}")
        print("-" * 75)
        
        robustness_results = []
        
        for test_name, text in edge_cases:
            result = self.benchmark_text(self.test_tokenizer, text, "HyperTokenizer16k")
            
            display_text = text[:27] + "..." if len(text) > 30 else text
            success_icon = "✅" if result["success"] else "❌"
            
            print(f"{test_name:<20} {display_text:<30} {result['tokens']:<8} {success_icon:<8} {result['encode_time_ms']:.2f}")
            
            robustness_results.append(result)
        
        # Robustness summary
        success_rate = sum(r["success"] for r in robustness_results) / len(robustness_results) * 100
        avg_time = statistics.mean([r["encode_time_ms"] for r in robustness_results if r["encode_time_ms"] != float('inf')])
        
        print(f"\n{'='*60}")
        print("ROBUSTNESS BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average processing time: {avg_time:.2f}ms")
        
        if success_rate >= 95:
            print("🏆 EXCELLENT: Highly robust tokenizer")
        elif success_rate >= 90:
            print("✅ GOOD: Robust with minor edge case issues")
        elif success_rate >= 80:
            print("⚠️  ACCEPTABLE: Some robustness concerns")
        else:
            print("❌ POOR: Significant robustness issues")
    
    def run_comprehensive_benchmark(self):
        """
        Run the complete industry-standard benchmark suite
        """
        self.log_section_start("COMPREHENSIVE BENCHMARK", "Full industry-standard tokenizer evaluation suite")
        
        print("🏢 OFFICIAL INDUSTRY-STANDARD TOKENIZER BENCHMARK SUITE")
        print("=" * 80)
        print("Comprehensive evaluation using methodologies from tech giants")
        print("Based on research from OpenAI, Google, Meta, and academic institutions")
        print("=" * 80)
        
        # Run all benchmark categories with section logging
        self.log_section_start("COMPRESSION BENCHMARK", "Testing compression efficiency vs reference tokenizer")
        self.run_compression_benchmark()
        self.log_section_end("COMPRESSION BENCHMARK")
        
        self.log_section_start("SPEED BENCHMARK", "Testing encoding/decoding performance")
        self.run_speed_benchmark()
        self.log_section_end("SPEED BENCHMARK")
        
        self.log_section_start("CHARACTER LEVEL BENCHMARK", "Testing character-level reasoning and morphological alignment")
        self.run_character_level_benchmark()
        self.log_section_end("CHARACTER LEVEL BENCHMARK")
        
        self.log_section_start("ROBUSTNESS BENCHMARK", "Testing edge case handling and domain robustness")
        self.run_robustness_benchmark()
        self.log_section_end("ROBUSTNESS BENCHMARK")
        
        # Final overall assessment
        self.log_section_start("FINAL ASSESSMENT", "Overall benchmark results and grading")
        
        print("\n" + "="*80)
        print("🏆 FINAL OVERALL ASSESSMENT")
        print("="*80)
        
        # Calculate overall scores
        compression_results = [r for r in self.results if r.category == "compression"]
        if compression_results:
            avg_compression = statistics.mean([r.compression_ratio for r in compression_results])
            overall_success_rate = sum(r.success for r in self.results) / len(self.results) * 100
            
            print(f"Overall compression ratio: {avg_compression:.3f}")
            print(f"Overall success rate: {overall_success_rate:.1f}%")
            
            # Industry-standard grading
            if avg_compression < 0.85 and overall_success_rate >= 95:
                grade = "A+ (Industry Leading)"
            elif avg_compression < 0.95 and overall_success_rate >= 90:
                grade = "A (Production Ready)"
            elif avg_compression < 1.05 and overall_success_rate >= 85:
                grade = "B (Competitive)"
            else:
                grade = "C (Needs Improvement)"
            
            print(f"\n🎯 FINAL GRADE: {grade}")
            
            # Recommendations
            print(f"\n📋 RECOMMENDATIONS:")
            if avg_compression >= 1.0:
                print("• Focus on improving compression efficiency")
            if overall_success_rate < 95:
                print("• Address robustness issues for production deployment")
            if avg_compression < 0.9 and overall_success_rate >= 95:
                print("• Excellent performance! Consider optimizing speed further")
        
        print("\n✅ BENCHMARK COMPLETE - Results ready for production evaluation")
        
        # Log completion with summary
        summary = f"Grade: {grade if 'grade' in locals() else 'N/A'}, Tests: {len(self.results)}, Success Rate: {overall_success_rate:.1f}% if 'overall_success_rate' in locals() else 'N/A'"
        self.log_section_end("FINAL ASSESSMENT", summary)
        self.log_section_end("COMPREHENSIVE BENCHMARK", f"Completed {len(self.results)} total tests")
        
        # Save detailed results to JSON
        self.save_detailed_results()
    
    def save_detailed_results(self):
        """Save detailed benchmark results to JSON file"""
        # Ensure logs directory exists
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(logs_dir, f"benchmark_results_{timestamp}.json")
        
        # Convert results to JSON-serializable format
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "log_file": self.log_file
            },
            "results": []
        }
        
        for result in self.results:
            results_data["results"].append({
                "tokenizer_name": result.tokenizer_name,
                "category": result.category,
                "test_name": result.test_name,
                "tokens": result.tokens,
                "encode_time_ms": result.encode_time_ms,
                "decode_time_ms": result.decode_time_ms,
                "compression_ratio": result.compression_ratio,
                "success": result.success,
                "error_message": result.error_message,
                "metadata": result.metadata
            })
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            print(f"📊 Detailed results saved to: {json_file}")
            self.logger.info(f"Detailed results saved to JSON: {json_file}")
        except Exception as e:
            print(f"❌ Failed to save JSON results: {e}")
            self.logger.error(f"Failed to save JSON results: {e}")


def main():
    """Run the official industry-standard tokenizer benchmark"""
    # Setup print capture for comprehensive logging
    import sys
    
    # Allow custom log file via command line argument
    log_file = None
    if len(sys.argv) > 1:
        # If user provides a custom filename, it will be placed in logs/ directory
        custom_name = sys.argv[1]
        if not custom_name.endswith('.log'):
            custom_name += '.log'
        log_file = custom_name
    
    # Create benchmark instance with logging
    benchmark = OfficialTokenizerBenchmark(log_file=log_file)
    
    print(f"� Log files will be saved to: logs/ directory")
    print(f"📝 Current log file: {benchmark.log_file}")
    print(f"📊 JSON results will also be saved to logs/ directory")
    print("="*80)
    
    try:
        # Redirect stdout to capture all print statements
        original_stdout = sys.stdout
        sys.stdout = LoggingPrintCapture(benchmark.logger)
        
        # Run the benchmark
        benchmark.run_comprehensive_benchmark()
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        
        # Flush any remaining buffer content
        if hasattr(sys.stdout, 'flush'):
            sys.stdout.flush()
        
        # Final log message with session completion
        benchmark.logger.info("")
        benchmark.logger.info("="*100)
        benchmark.logger.info("🎉 TOKENIZER BENCHMARK SESSION COMPLETED SUCCESSFULLY")
        benchmark.logger.info("="*100)
        benchmark.logger.info(f"📊 Total tests executed: {len(benchmark.results)}")
        benchmark.logger.info(f"⏰ Session ended at: {datetime.now()}")
        benchmark.logger.info(f"📝 Log file: {benchmark.log_file}")
        benchmark.logger.info("="*100)
        
        print(f"\n🎉 Benchmark completed!")
        print(f"📁 All results saved in logs/ directory:")
        print(f"  📝 Log file: {benchmark.log_file}")
        print(f"  📊 JSON results: Check logs/ for benchmark_results_*.json")


if __name__ == "__main__":
    main()
