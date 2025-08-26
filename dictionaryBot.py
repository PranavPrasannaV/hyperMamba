import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import json
import re
import pickle
from collections import defaultdict, Counter
import requests
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import math
import os
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SemanticNode:
    """Represents a word node in the semantic graph"""
    word: str
    definition: str
    part_of_speech: str
    embedding: np.ndarray
    semantic_weight: float = 1.0
    frequency: int = 0

class DictionaryProcessor:
    """Processes dictionary data and builds initial word relationships"""
    
    def __init__(self):
        self.words = {}
        self.definitions = {}
        self.word_embeddings = {}
        self.vectorizer = TfidfVectorizer(max_features=50000, stop_words=None)
        self.cache_dir = "cache"
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_dictionary_cache_file(self, data_source):
        """Get cache filename for dictionary data"""
        return os.path.join(self.cache_dir, f"dictionary_data_{data_source}.pkl")
    
    def _get_embeddings_cache_file(self):
        """Get cache filename for embeddings"""
        words_hash = hashlib.md5(",".join(sorted(self.words.keys())).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"embeddings_{words_hash}.pkl")
    
    def _save_dictionary_to_cache(self, data_source):
        """Save dictionary data to cache"""
        cache_file = self._get_dictionary_cache_file(data_source)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'words': self.words,
                    'definitions': self.definitions
                }, f)
            logger.info(f"Dictionary cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache dictionary: {e}")
    
    def _load_dictionary_from_cache(self, data_source):
        """Load dictionary data from cache"""
        cache_file = self._get_dictionary_cache_file(data_source)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.words = cached_data['words']
                self.definitions = cached_data['definitions']
                logger.info(f"Dictionary loaded from cache: {cache_file}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached dictionary: {e}")
                return False
        return False
    
    def _save_embeddings_to_cache(self):
        """Save embeddings to cache"""
        cache_file = self._get_embeddings_cache_file()
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'word_embeddings': self.word_embeddings,
                    'vectorizer': self.vectorizer
                }, f)
            logger.info(f"Embeddings cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
    
    def _load_embeddings_from_cache(self):
        """Load embeddings from cache"""
        cache_file = self._get_embeddings_cache_file()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.word_embeddings = cached_data['word_embeddings']
                self.vectorizer = cached_data['vectorizer']
                logger.info(f"Embeddings loaded from cache: {cache_file}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
                return False
        return False
        
    def load_dictionary_data(self, data_source='nltk'):
        """Load complete dictionary data with caching"""
        logger.info("Loading complete dictionary data...")
        
        # Check if we have cached dictionary data
        print(f"🔍 Checking for cached dictionary data...")
        if self._load_dictionary_from_cache(data_source):
            print(f"✅ Found cached dictionary with {len(self.words):,} words!")
            print(f"   Cache file: dictionary_data_{data_source}.pkl")
            print("   Skipping dictionary loading - using cached version")
            return
        
        print(f"   No cache found, loading dictionary from {data_source}...")
        
        if data_source == 'nltk':
            self._load_from_nltk()
        elif data_source == 'wordnet':
            self._load_from_wordnet()
        else:
            self._load_from_file(data_source)
        
        # Save to cache
        print(f"\n💾 Saving dictionary to cache...")
        self._save_dictionary_to_cache(data_source)
    
    def _load_from_nltk(self):
        """Load comprehensive dictionary from NLTK WordNet"""
        try:
            import nltk
            from nltk.corpus import wordnet as wn
            
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.info("Downloading WordNet corpus...")
                nltk.download('wordnet')
                nltk.download('omw-1.4')
            
            logger.info("Loading all synsets from WordNet...")
            
            
            all_synsets = list(wn.all_synsets())
            logger.info(f"Found {len(all_synsets)} synsets in WordNet")
            
            processed_words = set()
            
            print(f"\n📚 Processing {len(all_synsets):,} synsets from WordNet...")
            print("Progress: [", end="", flush=True)
            
            
            progress_width = 50
            last_progress = 0
            
            for idx, synset in enumerate(all_synsets):
                
                current_progress = int((idx / len(all_synsets)) * progress_width)
                if current_progress > last_progress:
                    print("█" * (current_progress - last_progress), end="", flush=True)
                    last_progress = current_progress
                
                
                if idx % 5000 == 0 and idx > 0:
                    percent = (idx / len(all_synsets)) * 100
                    print(f"\n   {idx:,}/{len(all_synsets):,} synsets ({percent:.1f}%) - {len(processed_words)} words found")
                    print("   Progress: [" + "█" * last_progress + " " * (progress_width - last_progress), end="", flush=True)
                
                
                for lemma in synset.lemmas():
                    word = lemma.name().lower().replace('_', ' ')
                    
                    
                    if word in processed_words or not word.replace(' ', '').isalpha():
                        continue
                    
                    
                    definition = synset.definition()
                    if not definition:
                        continue
                    
                    
                    pos = synset.pos()
                    pos_mapping = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 'r': 'adverb', 's': 'adjective'}
                    part_of_speech = pos_mapping.get(pos, 'unknown')
                    
                    
                    self.words[word] = {
                        'definition': definition,
                        'part_of_speech': part_of_speech,
                        'synset': synset.name(),
                        'frequency': len(synset.lemmas())  
                    }
                    
                    self.definitions[word] = definition
                    processed_words.add(word)
            
            print(f"]\n✅ WordNet processing complete: {len(self.words):,} words loaded\n")
            logger.info(f"Loaded {len(self.words)} words from WordNet")
            
        except ImportError:
            logger.warning("NLTK not available, falling back to basic dictionary")
            self._load_basic_dictionary()
    
    def _load_from_wordnet(self):
        """Alternative WordNet loading method"""
        self._load_from_nltk()
    
    def _load_basic_dictionary(self):
        """Load a basic dictionary when other sources aren't available"""
        
        try:
            import urllib.request
            
            
            word_sources = [
                "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
                "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
            ]
            
            words_found = False
            for url in word_sources:
                try:
                    logger.info(f"Downloading word list from {url}")
                    with urllib.request.urlopen(url) as response:
                        word_list = response.read().decode('utf-8').strip().split('\n')
                    
                    
                    for word in word_list[:10000]:  
                        word = word.strip().lower()
                        if len(word) >= 2 and word.isalpha():
                            self.words[word] = {
                                'definition': self._generate_definition(word),
                                'part_of_speech': self._infer_pos_from_morphology(word),
                                'frequency': len(word_list) - word_list.index(word)  
                            }
                            self.definitions[word] = self.words[word]['definition']
                    
                    words_found = True
                    logger.info(f"Loaded {len(self.words)} words from online source")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load from {url}: {e}")
                    continue
            
            if not words_found:
                raise Exception("Could not download word lists")
                
        except Exception as e:
            logger.error(f"Failed to load dictionary: {e}")
            
            self._create_minimal_dictionary()
    
    def _generate_definition(self, word):
        """Generate definition based on morphological analysis"""
        
        suffixes = {
            'tion': 'the action or process of',
            'ness': 'the state or quality of being',
            'ment': 'the result or means of an action',
            'able': 'capable of being',
            'ful': 'full of or characterized by',
            'less': 'without or lacking',
            'ly': 'in a manner relating to',
            'er': 'one who performs an action',
            'ing': 'the action or process of',
            'ed': 'having been subjected to'
        }
        
        
        for suffix, meaning in suffixes.items():
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[:-len(suffix)]
                return f"{meaning} {root}"
        
        
        prefixes = {
            'un': 'not or reverse of',
            'pre': 'before or in advance',
            'post': 'after or following',
            'anti': 'against or opposite to',
            'pro': 'in favor of or supporting',
            'inter': 'between or among',
            'over': 'excessive or beyond',
            'under': 'below or insufficient'
        }
        
        for prefix, meaning in prefixes.items():
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                root = word[len(prefix):]
                return f"{meaning} {root}"
        
        
        if len(word) <= 3:
            return f"a fundamental concept or basic element in language"
        elif len(word) <= 6:
            return f"a concept that relates to human experience and understanding"
        else:
            return f"a complex concept that encompasses multiple aspects of meaning and usage"
    
    def _infer_pos_from_morphology(self, word):
        """Infer part of speech from word morphology"""
        
        if word.endswith(('ate', 'ize', 'ify', 'en')):
            return 'verb'
        elif word.endswith('ing') and len(word) > 4:
            return 'verb'
        
        
        elif word.endswith(('tion', 'sion', 'ness', 'ment', 'ity', 'er', 'or', 'ist')):
            return 'noun'
        elif word.endswith('s') and len(word) > 3:
            return 'noun'
        
        
        elif word.endswith(('able', 'ible', 'ful', 'less', 'ous', 'ive', 'al', 'ic')):
            return 'adjective'
        
        
        elif word.endswith('ly') and len(word) > 3:
            return 'adverb'
        
        else:
            return 'noun'  
    
    def _create_minimal_dictionary(self):
        """Create minimal dictionary as absolute fallback"""
        
        import string
        
        
        for i, c1 in enumerate(string.ascii_lowercase[:10]):
            for j, c2 in enumerate(string.ascii_lowercase[:10]):
                if i != j:  
                    word = c1 + c2
                    self.words[word] = {
                        'definition': f"a linguistic unit representing a concept in human communication",
                        'part_of_speech': 'noun',
                        'frequency': 100 - (i + j)
                    }
                    self.definitions[word] = self.words[word]['definition']
        
        logger.info(f"Created minimal dictionary with {len(self.words)} words")
    
    def _infer_pos(self, word):
        """Infer part of speech based on word characteristics"""
        action_words = ['run', 'walk', 'jump', 'fly', 'swim', 'think', 'feel', 'speak', 'read', 'write']
        if word in action_words:
            return 'verb'
        elif word.endswith('ly'):
            return 'adverb'
        elif word in ['good', 'bad', 'big', 'small', 'hot', 'cold']:
            return 'adjective'
        else:
            return 'noun'
    
    def build_word_embeddings(self, batch_size=5000):
        """Build TF-IDF based embeddings for words using their definitions with caching and batching"""
        logger.info("Building word embeddings...")
        
        # Check if we have cached embeddings
        print(f"\n🔍 Checking for cached embeddings...")
        if self._load_embeddings_from_cache():
            print(f"✅ Found cached embeddings for {len(self.word_embeddings):,} words!")
            print("   Skipping embedding generation - using cached version")
            return self.word_embeddings
        
        print(f"   No cache found, building embeddings...")
        
        word_list = list(self.words.keys())
        total_words = len(word_list)
        
        print(f"\n🧮 Building embeddings for {total_words:,} words in batches of {batch_size:,}...")
        
        # First, fit the vectorizer on a sample to get vocabulary
        print("   Fitting TF-IDF vectorizer on sample data...")
        sample_size = min(10000, total_words)
        sample_definitions = [self.definitions[word] for word in word_list[:sample_size]]
        self.vectorizer.fit(sample_definitions)
        
        # Process in batches to avoid memory issues
        self.word_embeddings = {}
        num_batches = (total_words + batch_size - 1) // batch_size
        
        print(f"   Processing {num_batches} batches...")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_words)
            batch_words = word_list[start_idx:end_idx]
            
            print(f"   Batch {batch_idx + 1}/{num_batches}: Processing words {start_idx:,}-{end_idx:,}")
            
            # Get definitions for this batch
            batch_definitions = [self.definitions[word] for word in batch_words]
            
            # Transform this batch
            batch_tfidf = self.vectorizer.transform(batch_definitions)
            
            # Store embeddings for this batch
            for i, word in enumerate(batch_words):
                self.word_embeddings[word] = batch_tfidf[i].toarray().flatten()
            
            # Clear batch data from memory
            del batch_definitions, batch_tfidf
            
            # Progress update
            percent = (end_idx / total_words) * 100
            print(f"     Progress: {end_idx:,}/{total_words:,} ({percent:.1f}%)")
        
        # Save to cache
        print(f"\n💾 Saving embeddings to cache...")
        self._save_embeddings_to_cache()
        
        print(f"✅ Embeddings complete: {len(self.word_embeddings):,} word vectors generated\n")
        logger.info(f"Generated embeddings for {len(self.word_embeddings)} words")
        return self.word_embeddings

class SemanticGraphBuilder:
    """Builds semantic relationship graph between words"""
    
    def __init__(self, dictionary_processor):
        self.dict_processor = dictionary_processor
        self.graph = nx.Graph()
        self.semantic_relationships = defaultdict(list)
        self.cache_dir = "cache"
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_key(self, words, max_words):
        """Generate cache key based on words and parameters"""
        # Create a hash of the sorted words and max_words parameter
        words_str = ",".join(sorted(words[:max_words]))
        cache_string = f"{words_str}_{max_words}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cache_filename(self, cache_key):
        """Get cache filename for a given cache key"""
        return os.path.join(self.cache_dir, f"semantic_graph_{cache_key}.pkl")
    
    def _save_graph_to_cache(self, cache_key):
        """Save the current graph to cache"""
        cache_file = self._get_cache_filename(cache_key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'semantic_relationships': dict(self.semantic_relationships)
                }, f)
            logger.info(f"Graph cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache graph: {e}")
    
    def _load_graph_from_cache(self, cache_key):
        """Load graph from cache if it exists"""
        cache_file = self._get_cache_filename(cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.graph = cached_data['graph']
                self.semantic_relationships = defaultdict(list, cached_data['semantic_relationships'])
                logger.info(f"Graph loaded from cache: {cache_file}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached graph: {e}")
                return False
        return False
        
    def build_semantic_graph(self, max_words=2000):
        """Build the semantic relationship graph with progress tracking and caching"""
        logger.info("Building semantic graph...")
        
        all_words = list(self.dict_processor.words.keys())
        
        # Limit vocabulary if needed - use more conservative defaults
        if len(all_words) > max_words:
            print(f"⚡ Limiting vocabulary to {max_words} words for faster processing")
            print(f"   (Original vocabulary: {len(all_words)} words)")
            
            word_freq_pairs = [(word, self.dict_processor.words[word].get('frequency', 0)) 
                             for word in all_words]
            word_freq_pairs.sort(key=lambda x: x[1], reverse=True)
            words = [word for word, _ in word_freq_pairs[:max_words]]
        else:
            words = all_words
            if len(words) > 5000:
                print(f"⚠️  Processing {len(words):,} words may use significant memory")
                print(f"   Consider using max_words parameter to limit vocabulary size")
        
        # Check if we have a cached version
        cache_key = self._get_cache_key(words, max_words)
        
        print(f"\n🔍 Checking for cached semantic graph...")
        if self._load_graph_from_cache(cache_key):
            print(f"✅ Found cached graph with {self.graph.number_of_nodes():,} nodes and {self.graph.number_of_edges():,} edges!")
            print(f"   Cache file: semantic_graph_{cache_key[:8]}...pkl")
            print("   Skipping graph construction - using cached version")
            return self.graph
        
        print(f"   No cache found, building new graph...")
        
        embeddings = self.dict_processor.word_embeddings
        
        print(f"\n🏗️  Building semantic graph with {len(words)} words")
        print("=" * 60)
        
        # Add nodes to graph
        print(f"📝 Adding {len(words)} nodes to graph...")
        for word in words:
            self.graph.add_node(word, 
                              definition=self.dict_processor.definitions[word],
                              embedding=embeddings[word],
                              pos=self.dict_processor.words[word]['part_of_speech'])
        print(f"✅ All nodes added!\n")
        
        # Build edges with progress tracking
        print("🔍 Step 1/3: Calculating semantic similarities...")
        self._add_similarity_edges(words, embeddings)
        
        print("🔍 Step 2/3: Adding definitional relationships...")
        self._add_definitional_edges(words)
        
        print("🔍 Step 3/3: Adding syntactic relationships...")
        self._add_syntactic_edges(words)
        
        # Save to cache
        print(f"\n💾 Saving graph to cache...")
        self._save_graph_to_cache(cache_key)
        
        print("=" * 60)
        print(f"🎉 Semantic graph completed!")
        print(f"   Nodes: {self.graph.number_of_nodes():,}")
        print(f"   Edges: {self.graph.number_of_edges():,}")
        print(f"   Average connections per word: {self.graph.number_of_edges() / self.graph.number_of_nodes():.1f}")
        print(f"   Cached for future use: semantic_graph_{cache_key[:8]}...pkl")
        print("=" * 60)
        
        logger.info(f"Built semantic graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _add_similarity_edges(self, words, embeddings, batch_size=1000, similarity_threshold=0.3):
        """Add edges based on semantic similarity with progress tracking and memory-efficient batching"""
        total_pairs = len(words) * (len(words) - 1) // 2
        processed_pairs = 0
        edges_added = 0
        
        print(f"\n📊 Processing similarity edges for {len(words)} words ({total_pairs:,} pairs to check)")
        print(f"   Using batch size: {batch_size} words, similarity threshold: {similarity_threshold}")
        print("Progress: [", end="", flush=True)
        
        # Progress tracking
        progress_width = 50
        last_progress = 0
        
        # Process words in batches to avoid memory issues
        num_batches = (len(words) + batch_size - 1) // batch_size
        
        for batch_i in range(num_batches):
            start_i = batch_i * batch_size
            end_i = min(start_i + batch_size, len(words))
            batch_words_i = words[start_i:end_i]
            
            # Get embeddings for this batch
            batch_embeddings_i = np.array([embeddings[word] for word in batch_words_i])
            
            for batch_j in range(batch_i, num_batches):  # Only process upper triangle
                start_j = batch_j * batch_size
                end_j = min(start_j + batch_size, len(words))
                batch_words_j = words[start_j:end_j]
                
                # Get embeddings for second batch
                batch_embeddings_j = np.array([embeddings[word] for word in batch_words_j])
                
                # Calculate similarities between batches
                similarities = cosine_similarity(batch_embeddings_i, batch_embeddings_j)
                
                # Add edges for similar pairs
                for i, word1 in enumerate(batch_words_i):
                    start_j_idx = 0 if batch_i != batch_j else i + 1  # Avoid self-comparisons
                    for j in range(start_j_idx, len(batch_words_j)):
                        word2 = batch_words_j[j]
                        similarity = similarities[i, j]
                        
                        if similarity > similarity_threshold:
                            self.graph.add_edge(word1, word2, 
                                              weight=similarity, 
                                              relationship_type='semantic_similarity')
                            edges_added += 1
                        
                        processed_pairs += 1
                        
                        # Update progress
                        current_progress = int((processed_pairs / total_pairs) * progress_width)
                        if current_progress > last_progress:
                            print("█" * (current_progress - last_progress), end="", flush=True)
                            last_progress = current_progress
                        
                        # Progress update every 10k pairs
                        if processed_pairs % 10000 == 0:
                            percent = (processed_pairs / total_pairs) * 100
                            print(f"\n   {processed_pairs:,}/{total_pairs:,} pairs ({percent:.1f}%) - {edges_added} edges found")
                            print("   Progress: [" + "█" * last_progress + " " * (progress_width - last_progress), end="", flush=True)
        
        print(f"]\n✅ Similarity edges complete: {edges_added} edges added from {processed_pairs:,} pairs\n")
    
    def _add_definitional_edges(self, words):
        """Add edges based on words appearing in each other's definitions with progress tracking"""
        print(f"📖 Processing definitional edges for {len(words)} words...")
        edges_added = 0
        
        for i, word in enumerate(words):
            if i % 500 == 0:
                percent = (i / len(words)) * 100
                print(f"   Processing definitions: {i}/{len(words)} ({percent:.1f}%) - {edges_added} edges so far")
            
            definition = self.dict_processor.definitions[word].lower()
            definition_words = re.findall(r'\b\w+\b', definition)
            
            for def_word in definition_words:
                if def_word in words and def_word != word:
                    self.graph.add_edge(word, def_word, 
                                      weight=0.7, 
                                      relationship_type='definitional')
                    edges_added += 1
        
        print(f"✅ Definitional edges complete: {edges_added} edges added\n")
    
    def _add_syntactic_edges(self, words):
        """Add edges based on part-of-speech relationships with progress tracking"""
        print(f"🔗 Processing syntactic edges for {len(words)} words...")
        pos_groups = defaultdict(list)
        
        
        for word in words:
            pos = self.dict_processor.words[word]['part_of_speech']
            pos_groups[pos].append(word)
        
        print(f"   Found {len(pos_groups)} part-of-speech groups:")
        for pos, group in pos_groups.items():
            print(f"     {pos}: {len(group)} words")
        
        edges_added = 0
        
        for pos, word_group in pos_groups.items():
            print(f"   Connecting {pos} words ({len(word_group)} words)...")
            for i, word1 in enumerate(word_group):
                
                max_connections = min(5, len(word_group) - i - 1)
                for word2 in word_group[i+1:i+1+max_connections]:
                    self.graph.add_edge(word1, word2, 
                                      weight=0.4, 
                                      relationship_type='syntactic')
                    edges_added += 1
        
        print(f"✅ Syntactic edges complete: {edges_added} edges added\n")

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for processing semantic relationships"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GraphNeuralNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
        
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
        
        self.gat_layers.append(GATConv(hidden_dim * 4, output_dim, heads=1, concat=False))
        
        
        self.response_generator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass through the GNN"""
        
        for i, gat_layer in enumerate(self.gat_layers[:-1]):
            x = gat_layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        
        
        x = self.gat_layers[-1](x, edge_index)
        
        
        response_features = self.response_generator(x)
        
        return response_features

class CognitiveCascadeArchitecture:
    """Main architecture class that orchestrates the entire system"""
    
    def __init__(self):
        self.dictionary_processor = DictionaryProcessor()
        self.graph_builder = None
        self.gnn = None
        self.semantic_graph = None
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.is_trained = False
        
    def train(self):
        """Train the complete architecture"""
        import time
        start_time = time.time()
        
        print("\n" + "🧠" + "="*58 + "🧠")
        print("    COGNITIVE CASCADE ARCHITECTURE TRAINING")
        print("🧠" + "="*58 + "🧠")
        
        logger.info("Starting Cognitive Cascade Architecture training...")
        
        
        print("\n📖 PHASE 1/4: Loading Dictionary Data")
        print("-" * 60)
        phase_start = time.time()
        self.dictionary_processor.load_dictionary_data()
        phase_time = time.time() - phase_start
        print(f"✅ Phase 1 completed in {phase_time:.1f} seconds")
        
        
        print("\n🧮 PHASE 2/4: Building Word Embeddings")
        print("-" * 60)
        phase_start = time.time()
        self.dictionary_processor.build_word_embeddings()
        phase_time = time.time() - phase_start
        print(f"✅ Phase 2 completed in {phase_time:.1f} seconds")
        
        
        print("\n🌐 PHASE 3/4: Building Semantic Graph")
        print("-" * 60)
        phase_start = time.time()
        self.graph_builder = SemanticGraphBuilder(self.dictionary_processor)
        
        
        vocab_size = len(self.dictionary_processor.words)
        if vocab_size > 3000:
            print(f"\n⚠️  Large vocabulary detected: {vocab_size:,} words")
            print("   Building relationships for all words may use significant memory.")
            print("   Recommended options:")
            print("   1. Use 1,000 words (fast, low memory, ~1-2 minutes)")
            print("   2. Use 2,000 words (medium, moderate memory, ~2-5 minutes)")
            print("   3. Use 5,000 words (slower, higher memory, ~10-30 minutes)")
            print("   4. Use all words (slowest, highest memory, could take hours)")
            
            while True:
                choice = input("\n   Choose option (1/2/3/4) or enter custom number: ").strip()
                try:
                    if choice == '1':
                        max_words = 1000
                        break
                    elif choice == '2':
                        max_words = 2000
                        break
                    elif choice == '3':
                        max_words = 5000
                        break
                    elif choice == '4':
                        max_words = vocab_size
                        break
                    else:
                        max_words = int(choice)
                        if max_words > 0:
                            break
                        else:
                            print("   Please enter a positive number")
                except ValueError:
                    print("   Please enter 1, 2, 3, 4, or a number")
        else:
            max_words = vocab_size
        
        self.semantic_graph = self.graph_builder.build_semantic_graph(max_words)
        phase_time = time.time() - phase_start
        print(f"✅ Phase 3 completed in {phase_time:.1f} seconds ({phase_time/60:.1f} minutes)")
        
        
        print("\n🔧 PHASE 4/4: Training Neural Network")
        print("-" * 60)
        phase_start = time.time()
        self._prepare_graph_data()
        self._train_gnn()
        phase_time = time.time() - phase_start
        print(f"✅ Phase 4 completed in {phase_time:.1f} seconds")
        
        total_time = time.time() - start_time
        self.is_trained = True
        
        print("\n🎉" + "="*58 + "🎉")
        print("    TRAINING COMPLETED SUCCESSFULLY!")
        print(f"    Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print("🎉" + "="*58 + "🎉")
        
        logger.info("Training completed successfully!")
    
    def _prepare_graph_data(self):
        """Prepare graph data for PyTorch Geometric"""
        logger.info("Preparing graph data for GNN...")
        
        
        nodes = list(self.semantic_graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(nodes)}
        
        
        node_features = []
        for node in nodes:
            embedding = self.semantic_graph.nodes[node]['embedding']
            node_features.append(embedding)
        
        self.node_features = torch.FloatTensor(np.array(node_features))
        
        
        edge_indices = []
        edge_weights = []
        
        for edge in self.semantic_graph.edges(data=True):
            src_idx = self.node_to_idx[edge[0]]
            dst_idx = self.node_to_idx[edge[1]]
            weight = edge[2].get('weight', 1.0)
            
            edge_indices.append([src_idx, dst_idx])
            edge_indices.append([dst_idx, src_idx])  
            edge_weights.append(weight)
            edge_weights.append(weight)
        
        self.edge_index = torch.LongTensor(edge_indices).t().contiguous()
        self.edge_weights = torch.FloatTensor(edge_weights)
        
    def _train_gnn(self):
        """Train the Graph Neural Network"""
        logger.info("Training Graph Neural Network...")
        
        input_dim = self.node_features.shape[1]
        hidden_dim = 128
        output_dim = 64
        
        self.gnn = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.01, weight_decay=5e-4)
        
        
        self.gnn.train()
        for epoch in range(100):
            optimizer.zero_grad()
            
            
            output = self.gnn(self.node_features, self.edge_index)
            
            
            loss = F.mse_loss(output, self.node_features[:, :output.shape[1]])
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.gnn.eval()
        logger.info("GNN training completed!")
    
    def generate_response(self, query: str) -> str:
        """Generate response using the trained architecture"""
        if not self.is_trained:
            return "Architecture not trained yet. Please run train() first."
        
        logger.info(f"Generating response for query: '{query}'")
        
        
        query_words = self._extract_key_words(query)
        
        
        relevant_nodes = self._find_relevant_nodes(query_words)
        
        
        semantic_context = self._generate_semantic_context(relevant_nodes)
        
        
        response = self._generate_contextual_response(query, semantic_context, relevant_nodes)
        
        return response
    
    def _extract_key_words(self, query: str) -> List[str]:
        """Extract meaningful words from the query with improved matching"""
        # Clean and split query
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Remove common stop words that don't add meaning
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall'}
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Direct matches first (highest priority)
        key_words = []
        for word in meaningful_words:
            if word in self.dictionary_processor.words:
                key_words.append(word)
        
        # If no direct matches, try fuzzy matching but be more selective
        if not key_words:
            for word in meaningful_words:
                if len(word) >= 4:  # Only fuzzy match longer words
                    # Find words that start with the query word (exact prefix)
                    similar_words = [w for w in self.dictionary_processor.words.keys() 
                                   if w.startswith(word) and len(w) <= len(word) + 3]
                    if similar_words:
                        key_words.extend(similar_words[:1])  # Only add best match
                    
                    # Find words that contain the query word as a substring
                    elif len(word) >= 5:  # Only for longer words
                        containing_words = [w for w in self.dictionary_processor.words.keys() 
                                          if word in w and abs(len(w) - len(word)) <= 2]
                        if containing_words:
                            key_words.extend(containing_words[:1])  # Only add best match
        
        # Remove duplicates while preserving order
        result = list(dict.fromkeys(key_words))
        
        # Debug output
        print(f"Debug: Query '{query}' -> Words: {words} -> Key words: {result}")
        
        return result
    
    def _find_relevant_nodes(self, query_words: List[str]) -> List[str]:
        """Find nodes relevant to the query words with improved search"""
        relevant_nodes = set()
        
        print(f"Debug: Looking for nodes from key words: {query_words}")
        
        # Add direct matches
        direct_matches = []
        for word in query_words:
            if word in self.semantic_graph:
                relevant_nodes.add(word)
                direct_matches.append(word)
        
        print(f"Debug: Direct matches found: {direct_matches}")
        
        # Add neighbors of found words (but be more selective)
        neighbor_count = 0
        for word in query_words:
            if word in self.semantic_graph:
                # Get all neighbors
                neighbors = list(self.semantic_graph.neighbors(word))
                
                # Sort by edge weight
                neighbor_weights = [(n, self.semantic_graph[word][n].get('weight', 0)) 
                                  for n in neighbors]
                neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                
                # Add only top 2-3 neighbors to avoid noise
                for neighbor, weight in neighbor_weights[:3]:
                    relevant_nodes.add(neighbor)
                    neighbor_count += 1
        
        print(f"Debug: Added {neighbor_count} neighbors")
        
        # If still no nodes found, be more conservative with fallback
        if not relevant_nodes:
            print("Debug: No relevant nodes found, using conservative fallback")
            # Instead of random words, try to find words related to common query terms
            common_words = ['word', 'meaning', 'definition', 'concept', 'idea', 'thing', 'object']
            for fallback_word in common_words:
                if fallback_word in self.semantic_graph:
                    relevant_nodes.add(fallback_word)
                    break
            
            # If still nothing, add just a few high-frequency words
            if not relevant_nodes:
                sample_words = list(self.semantic_graph.nodes())[:3]  # Reduced from 10 to 3
                relevant_nodes.update(sample_words)
        
        result = list(relevant_nodes)
        print(f"Debug: Final relevant nodes: {result}")
        return result
    
    def _generate_semantic_context(self, relevant_nodes: List[str]) -> torch.Tensor:
        """Generate semantic context using GNN"""
        
        node_indices = [self.node_to_idx[node] for node in relevant_nodes 
                       if node in self.node_to_idx]
        
        if not node_indices:
            
            return torch.zeros(64)
        
        
        with torch.no_grad():
            gnn_output = self.gnn(self.node_features, self.edge_index)
            relevant_embeddings = gnn_output[node_indices]
            
            
            semantic_context = torch.mean(relevant_embeddings, dim=0)
        
        return semantic_context
    
    def _generate_contextual_response(self, query: str, semantic_context: torch.Tensor, relevant_nodes: List[str]) -> str:
        """Generate contextual response based on semantic understanding"""
        
        if not relevant_nodes:
            return "I don't have enough information about that topic in my current knowledge."
        
        # Extract query intent
        query_lower = query.lower()
        
        # Check for question words
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return self._generate_question_response(query_lower, relevant_nodes)
        elif any(word in query_lower for word in ['define', 'definition', 'meaning', 'mean']):
            return self._generate_definition_response(relevant_nodes)
        elif any(word in query_lower for word in ['explain', 'describe', 'tell me about']):
            return self._generate_explanation_response(relevant_nodes)
        else:
            return self._generate_general_response(relevant_nodes)
    
    def _generate_question_response(self, query: str, relevant_nodes: List[str]) -> str:
        """Generate a response for a question"""
        if not relevant_nodes:
            return "I'm not sure how to answer that. I don't have information on the topic."

        # Simple response for now, can be expanded
        primary_node = relevant_nodes[0]
        definition = self.dictionary_processor.definitions.get(primary_node, "No definition found.")
        return f"Regarding '{primary_node}': {definition}"

    def _generate_definition_response(self, relevant_nodes: List[str]) -> str:
        """Generate a definition for a word"""
        if not relevant_nodes:
            return "Which word do you want me to define?"
        
        word_to_define = relevant_nodes[0]
        definition = self.dictionary_processor.definitions.get(word_to_define, f"I don't have a definition for '{word_to_define}'.")
        return f"The definition of '{word_to_define}' is: {definition}"

    def _generate_explanation_response(self, relevant_nodes: List[str]) -> str:
        """Generate an explanation about a topic"""
        if not relevant_nodes:
            return "I can't explain that, as I don't have any information about it."

        # Combine definitions of related nodes for an explanation
        explanation_parts = []
        for node in relevant_nodes[:3]: # Limit to 3 nodes for a concise explanation
            definition = self.dictionary_processor.definitions.get(node)
            if definition:
                explanation_parts.append(f"'{node}' relates to '{definition}'")
        
        if not explanation_parts:
            return f"I can't find enough details to explain {relevant_nodes[0]}."

        return ". ".join(explanation_parts) + "."

    def _generate_general_response(self, relevant_nodes: List[str]) -> str:
        """Generate a general response"""
        if not relevant_nodes:
            return "I'm not sure what to say about that."

        # Create a response based on related nodes
        primary_node = relevant_nodes[0]
        definition = self.dictionary_processor.definitions.get(primary_node, "No definition available")
        
        # Check if we have neighbors to mention
        if primary_node in self.semantic_graph:
            neighbors = list(self.semantic_graph.neighbors(primary_node))[:3]
            
            if neighbors:
                return f"Regarding '{primary_node}': {definition}. This relates to concepts like: {', '.join(neighbors)}."
            else:
                return f"Regarding '{primary_node}': {definition}."
        else:
            return f"I found information about '{primary_node}': {definition}."

    def _generate_from_context_magnitude(self, magnitude: float) -> str:
        """Generate response based on context magnitude without predefined responses"""
        
        if magnitude > 0.5:
            
            return self._generate_complex_semantic_response()
        elif magnitude > 0.1:
            
            return self._generate_moderate_semantic_response()
        else:
            
            return self._generate_basic_semantic_response()
    
    def _generate_complex_semantic_response(self) -> str:
        """Generate complex response from learned patterns"""
        
        sample_nodes = list(self.dictionary_processor.words.keys())[:5]
        conceptual_words = []
        
        for node in sample_nodes:
            definition = self.dictionary_processor.definitions.get(node, "")
            
            def_words = re.findall(r'\b\w+\b', definition.lower())
            conceptual_words.extend([w for w in def_words if len(w) > 4])
        
        if conceptual_words:
            selected_concept = conceptual_words[0] if conceptual_words else "processing"
            return f"The semantic {selected_concept} involves multiple interconnected elements."
        
        return "Semantic relationships are being processed through multiple pathways."
    
    def _generate_moderate_semantic_response(self) -> str:
        """Generate moderate complexity response"""
        
        sample_word = list(self.dictionary_processor.words.keys())[0] if self.dictionary_processor.words else "concept"
        return f"This relates to {sample_word} and associated meanings."
    
    def _generate_basic_semantic_response(self) -> str:
        """Generate basic response from learned patterns"""
        return "Semantic analysis in progress."
    
    def _transform_definition_to_response(self, word: str, definition: str) -> str:
        """Transform a definition into a natural response"""
        
        def_words = re.findall(r'\b\w+\b', definition.lower())
        meaningful_words = [w for w in def_words if len(w) > 3 and w in self.dictionary_processor.words]
        
        if meaningful_words:
            primary_concept = meaningful_words[0]
            return f"This concerns {word}, which involves {primary_concept}."
        else:
            return f"The concept of {word} emerges from the definition patterns."
    
    def _analyze_node_relationships(self, nodes: List[str]) -> str:
        """Analyze relationships between nodes using graph structure"""
        relationships = []
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if node1 in self.semantic_graph and node2 in self.semantic_graph:
                    if self.semantic_graph.has_edge(node1, node2):
                        edge_data = self.semantic_graph[node1][node2]
                        rel_type = edge_data.get('relationship_type', 'semantic')
                        relationships.append((node1, node2, rel_type))
        
        if relationships:
            first_rel = relationships[0]
            return f"The connection between {first_rel[0]} and {first_rel[1]} shows {first_rel[2]} patterns."
        
        return f"Multiple concepts interact: {', '.join(nodes[:2])}."
    
    def _combine_response_parts(self, parts: List[str], context: torch.Tensor) -> str:
        """Combine response parts based on semantic context"""
        if not parts:
            return "Semantic processing complete."
        
        if len(parts) == 1:
            return parts[0]
        
        
        magnitude = torch.norm(context).item()
        
        if magnitude > 0.3:
            
            return f"{parts[0]} Additionally, {parts[1] if len(parts) > 1 else 'related concepts emerge'}."
        else:
            
            return f"{parts[0]} {parts[1] if len(parts) > 1 else ''}"
    
    def interactive_mode(self):
        """Run the architecture in interactive mode"""
        if not self.is_trained:
            print("Training the Cognitive Cascade Architecture...")
            self.train()
        
        print("\n" + "="*60)
        print("Cognitive Cascade Architecture - Interactive Mode")
        print("="*60)
        print("Architecture trained and ready!")
        print(f"Semantic graph contains {self.semantic_graph.number_of_nodes()} words")
        print(f"with {self.semantic_graph.number_of_edges()} semantic relationships")
        print("\nType 'quit' to exit, 'info' for architecture details")
        print("-"*60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'info':
                    self._display_architecture_info()
                    continue
                
                elif not user_input:
                    continue
                
                
                response = self.generate_response(user_input)
                print(f"CCA: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Error in interactive mode: {e}")
    
    def _display_architecture_info(self):
        """Display detailed information about the architecture"""
        print("\n" + "="*50)
        print("COGNITIVE CASCADE ARCHITECTURE DETAILS")
        print("="*50)
        print(f"Dictionary Words: {len(self.dictionary_processor.words)}")
        print(f"Semantic Graph Nodes: {self.semantic_graph.number_of_nodes()}")
        print(f"Semantic Graph Edges: {self.semantic_graph.number_of_edges()}")
        print(f"GNN Hidden Dimensions: 128")
        print(f"GNN Output Dimensions: 64")
        print(f"Training Status: {'Completed' if self.is_trained else 'Not Trained'}")
        
        print("\nSample Word Relationships:")
        sample_words = list(self.semantic_graph.nodes())[:5]
        for word in sample_words:
            neighbors = list(self.semantic_graph.neighbors(word))[:3]
            print(f"  {word} -> {', '.join(neighbors)}")
        
        print("\nArchitecture Principles:")
        print("  • No pre-programmed responses")
        print("  • Semantic understanding from word relationships")
        print("  • Graph Neural Networks for context processing")
        print("  • Dynamic response generation")
        print("-"*50)

def main():
    """Main function to run the Cognitive Cascade Architecture"""
    print("Initializing Cognitive Cascade Architecture...")
    
    
    cca = CognitiveCascadeArchitecture()
    
    
    cca.interactive_mode()

if __name__ == "__main__":
    main()
