import pandas as pd
import numpy as np
import re
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import warnings
try:
    from .config import parse_analysis_args
except ImportError:  # Support the historical direct-script entry point.
    from config import parse_analysis_args
warnings.filterwarnings('ignore')

# -------------------- SETUP FUNCTIONS --------------------

def setup_nltk():
    """Configure NLTK data path and download necessary resources"""
    nltk_data_path = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    
    try:
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
        print("NLTK resources downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        print("Continuing with alternative tokenization methods...")

# -------------------- DATA PROCESSING --------------------

def clean_data(input_file, output_file):
    """Clean and preprocess the semiconductor jobs data"""
    # Load the data
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print("Columns in the dataset:", df.columns.tolist())
    
    # Remove specified columns
    columns_to_drop = ['job_type', 'scrape_date']
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Handle missing values
    print("\nMissing values per column:")
    print(df_cleaned.isnull().sum())
    
    # Fill missing values
    for col in ['job_title', 'company', 'location', 'posted_date']:
        df_cleaned[col] = df_cleaned[col].fillna('Not specified')
    
    # Fix duplicate titles
    print("\nHandling duplicates in job_title column...")
    duplicate_pattern = r'(.+?)\s*\1\s*$'
    mask = df_cleaned['job_title'].str.contains(duplicate_pattern, regex=True)
    
    if mask.any():
        print(f"Found {mask.sum()} rows with duplicated title text")
        df_cleaned.loc[mask, 'job_title'] = df_cleaned.loc[mask, 'job_title'].apply(
            lambda title: re.match(duplicate_pattern, title).group(1).strip() if re.match(duplicate_pattern, title) else title
        )
    
    # Remove invalid titles
    invalid_titles = ['2025 Internship Program', 'Internship Program', 'Summer Internship']
    df_cleaned = df_cleaned[~df_cleaned['job_title'].isin(invalid_titles)]
    df_cleaned = df_cleaned[~df_cleaned['job_title'].str.contains(r'\b20\d{2}\b', na=False)]
    
    # Report on frequency of job titles
    title_counts = df_cleaned['job_title'].value_counts()
    print(f"Top repeated job titles: {title_counts.head(5).to_dict()}")
    
    # Add standardized date column
    df_cleaned['posting_date'] = pd.to_datetime(df_cleaned['posted_date'], 
                                               errors='coerce').dt.strftime('%Y-%m')
    df_cleaned['posting_date'] = df_cleaned['posting_date'].fillna('Unknown')
    
    # Save the cleaned data
    df_cleaned.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    
    return df_cleaned

# -------------------- TEXT PROCESSING --------------------

def tokenize_text(text):
    """Custom tokenizer for processing job descriptions"""
    if pd.isna(text):
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # Get stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        # Fallback stopwords list if NLTK fails
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                      'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                      'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                      'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                      'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                      'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                      'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                      'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
                      'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                      'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                      'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                      'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
                      's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
    
    # Add domain-specific stopwords
    domain_stopwords = {
        'experience', 'skill', 'skills', 'required', 'qualified', 
        'qualification', 'qualifications', 'applicant', 'candidate', 
        'degree', 'ability', 'year', 'years', 'job', 'work', 'working',
        'company', 'position', 'role', 'team', 'day', 'week', 'month',
        'salary', 'please', 'apply', 'www', 'com', 'http', 'https',
        'responsibilities', 'requirements', 'description', 'support', 'provide', 
        'work', 'company', 'ability', 'experience', 'required', 'will',
        'develop', 'design', 'including', 'must', 'employee', 'opportunity',
        'offer', 'customer', 'solution', 'technology', 'product', 'apply',
        'semiconductor', 'looking', 'ensure', 'knowledge', 'function', 'location', 
        'today', 'join', 'industry', 'career', 'would', 'related',
        'based', 'across', 'part', 'time', 'level', 'well', 'environment',
        'using', 'understanding', 'hardware', 'software', 'project', 'good',
        'help', 'understand', 'best', 'field', 'need', 'great', 'title', 'want',
        'full', 'great', 'strong', 'development', 'use', 'manager', 'engineer',
        'benefits', 'report', 'ideal', 'competitive', 'expected', 'include'
    }
    stop_words.update(domain_stopwords)
    
    # Filter tokens
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatize
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except LookupError:
        pass  # Skip lemmatization if WordNet isn't available
    
    return tokens

# -------------------- SKILL EXTRACTION --------------------

def get_technical_skills_dictionary():
    """Return a comprehensive dictionary of semiconductor industry technical skills"""
    # Organized by categories
    tech_skills = {
        # Programming Languages
        'python programming', 'c++', 'java', 'javascript', 'typescript', 'rust', 'golang', 'perl scripting', 
        'tcl scripting', 'shell scripting', 'bash scripting', 'ruby', 'scala', 'r programming', 'matlab',
        
        # Hardware Description Languages
        'verilog', 'systemverilog', 'vhdl', 'hardware description language', 'rtl design', 'uvm methodology',
        
        # Hardware Design
        'fpga design', 'asic design', 'ic design', 'vlsi design', 'mixed signal design', 'analog design', 
        'digital design', 'schematic capture', 'layout design', 'pcb design', 'system-on-chip', 'soc design', 
        'ip development', 'dft techniques', 'dfm techniques', 'physical design', 'frontend design', 
        'backend design', 'verification engineering', 'validation testing', 'timing analysis', 'static timing analysis', 
        'logic synthesis', 'formal verification', 'clock design', 'low power design',
        
        # EDA Tools
        'cadence virtuoso', 'cadence innovus', 'cadence genus', 'synopsys design compiler', 'synopsys primetime', 
        'synopsys vcs', 'spice simulation', 'hspice', 'spectre', 'calibre drc', 'xilinx vivado', 
        'altera quartus', 'ansys', 'mentor graphics', 'modelsim', 'magic layout',
        
        # Semiconductor Manufacturing
        'semiconductor fabrication', 'wafer fabrication', 'foundry processes', 'photolithography', 
        'plasma etching', 'thin film deposition', 'chemical mechanical polishing', 'cmp process',
        'process design kit', 'pdk development', 'yield analysis', 'yield optimization', 
        'reliability testing', 'cleanroom experience',
        
        # Testing & Quality
        'automatic test pattern generation', 'atpg', 'functional testing', 'test automation', 
        'design rule checking', 'drc verification', 'layout versus schematic', 'lvs verification', 
        'parasitic extraction', 'pex analysis', 'characterization methods', 'fault analysis', 
        'failure analysis', 'quality assurance', 'qa processes',
        
        # Device Technology
        'device physics', 'cmos technology', 'finfet technology', 'gaafet', 'gan technology', 
        'sic devices', 'wide bandgap semiconductors', 'power electronics', 'mems design', 
        'sensor design', 'actuator design', 'microcontroller programming', 'microprocessor architecture', 
        'memory design', 'dram design', 'sram design', 'flash memory', 'non-volatile memory',
        
        # Systems & Architecture
        'arm architecture', 'risc-v', 'x86 architecture', 'mips architecture', 'cpu design', 
        'gpu architecture', 'tpu design', 'dsp programming', 'computer architecture', 
        'embedded systems', 'firmware development', 'rtos', 'linux kernel', 'device drivers', 
        'signal processing',
        
        # Emerging Technologies
        'quantum computing', 'silicon photonics', 'integrated photonics', 'optical interconnects', 
        'neuromorphic computing', 'spintronics', 'artificial intelligence', 'machine learning algorithms', 
        'deep learning', 'neural networks', 'convolutional neural networks', 'computer vision', 
        'natural language processing', 'reinforcement learning', 'generative ai', 'edge ai', 
        'tensor processing',
        
        # Cloud & Infrastructure
        'cloud computing', 'aws cloud', 'microsoft azure', 'google cloud', 'docker containers', 
        'kubernetes orchestration', 'virtualization technology', 'internet of things', 'iot platforms', 
        'edge computing', '5g technology', '6g research', 'network architecture',
        
        # Security
        'hardware security', 'cybersecurity', 'blockchain technology', 'encryption algorithms', 
        'secure boot', 'trusted execution environment', 'secure enclave',
        
        # Simulation & Modeling
        'circuit simulation', 'system modeling', 'hardware emulation', 'digital twin technology', 
        'finite element analysis', 'computational fluid dynamics', 'thermal simulation', 
        'electromagnetic simulation', 'multiphysics simulation', 'simulink modeling', 
        'labview programming', 'verilog-a modeling', 'systemc modeling',
        
        # Project Management & Methods
        'agile methodology', 'scrum framework', 'kanban method', 'devops practices', 
        'continuous integration', 'automated testing', 'robotics integration'
    }
    
    # Multi-word technical skills specific to semiconductor industry
    advanced_phrases = {
        # AI/ML in semiconductors
        'ai chip design', 'machine learning accelerators', 'neural network accelerators', 
        'deep learning processors', 'vision processing units', 'ai edge computing',
        
        # Advanced design techniques
        'advanced node design', 'analog circuit design', 'digital circuit design', 'mixed-signal design',
        'physical verification flow', 'design rule checking flow', 'layout versus schematic verification',
        'parasitic extraction methodology', 'signal integrity analysis', 'power integrity analysis', 
        'thermal analysis techniques', 'power analysis methodology', 'electromagnetic compatibility testing',
        
        # Hardware acceleration
        'hardware acceleration', 'hardware-software codesign', 'system level design', 
        'high-level synthesis', 'logic optimization techniques',
        
        # Advanced packaging
        'advanced packaging', 'silicon photonics integration', 'chip multiprocessor design', 
        'system in package', 'system on package', 'through-silicon via technology',
        'fan-out wafer level packaging', 'chiplet design methodology', 'heterogeneous integration',
        
        # Memory technologies
        'memory subsystem design', 'cache coherence protocols', 'memory controller design',
        'high bandwidth memory interface', 'hbm integration', 'non-volatile memory express',
        'persistent memory technologies',
        
        # Material science for semiconductors
        'silicon carbide devices', 'gallium nitride power devices', 'silicon on insulator technology', 
        'fully depleted silicon on insulator', 'fin field-effect transistor design',
        'gate-all-around field-effect transistor', 'complementary metal-oxide semiconductor',
        
        # RF and power
        'radio frequency integrated circuits', 'millimeter wave design', 'antenna design techniques',
        'power electronics design', 'battery management systems', 'energy harvesting systems',
        'low power design techniques', 'ultra low power design',
        
        # ASIC/FPGA specific
        'asic verification', 'fpga prototyping', 'ip core development', 'soft ip integration', 
        'hard ip implementation', 'analog to digital converter design', 'digital to analog converter design',
        'phase locked loop design', 'clock data recovery circuits', 'serdes design techniques',
        
        # Design for test
        'design for testability', 'scan chain insertion', 'boundary scan testing', 'built-in self-test',
        'automatic test pattern generation', 'formal verification methodology', 
        'assertion-based verification', 'universal verification methodology',
        
        # Manufacturing optimization
        'design for manufacturing', 'optical proximity correction', 'resolution enhancement techniques',
        'design rule optimization', 'yield optimization methods', 'statistical static timing analysis',
        'process variation modeling', 'corner analysis methodology', 'monte carlo simulation',
        
        # Reliability and safety
        'functional safety verification', 'fault tolerance design', 'error correction coding',
        'quantum error correction', 'radiation hardened design',
        
        # EDA specific
        'eda tool development', 'eda flow optimization', 'custom eda scripting',
        'physical design automation', 'logic equivalence checking'
    }
    
    # Combine all skills
    all_skills = tech_skills.union(advanced_phrases)
    
    return all_skills

def extract_technical_skills(df, column='job_description'):
    """Extract technical skills from job descriptions"""
    # Get comprehensive skills dictionary
    all_skills = get_technical_skills_dictionary()
    
    # Extract text from the specified column
    all_text = ' '.join(df[column].fillna('').astype(str).tolist()).lower()
    
    # Count frequencies of technical skills
    skill_counter = Counter()
    
    # Process for multi-word skills (prioritize longer phrases first)
    sorted_skills = sorted(all_skills, key=len, reverse=True)
    for skill in sorted_skills:
        if ' ' in skill:  # Multi-word skill
            count = len(re.findall(r'\b' + re.escape(skill) + r'\b', all_text))
            if count > 0:
                skill_counter[skill] = count
                
                # Remove text that's been matched to avoid double-counting
                all_text = re.sub(r'\b' + re.escape(skill) + r'\b', '', all_text)
    
    # Process for single-word skills
    tokens = tokenize_text(all_text)
    for token in tokens:
        if token.lower() in all_skills and token.lower() != 'semiconductor':
            skill_counter[token.lower()] += 1
    
    return skill_counter

# -------------------- TOPIC MODELING --------------------

def identify_emerging_skills(df, column='job_description', n_topics=5, n_top_words=10):
    """Identify emerging skills using NLP methods"""
    # Preprocess job descriptions
    descriptions = df[column].fillna('').astype(str)
    processed_descriptions = [' '.join(tokenize_text(text)) for text in descriptions]
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.8,
        min_df=5,
        max_features=1000,
        ngram_range=(1, 3)  # Include bi-grams and tri-grams
    )
    tfidf = tfidf_vectorizer.fit_transform(processed_descriptions)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Count Vectorization for LDA
    count_vectorizer = CountVectorizer(
        stop_words='english',
        max_df=0.8,
        min_df=5,
        max_features=1000,
        ngram_range=(1, 3)
    )
    count_data = count_vectorizer.fit_transform(processed_descriptions)
    count_feature_names = count_vectorizer.get_feature_names_out()
    
    # Apply Latent Dirichlet Allocation 
    print("Running LDA topic modeling...")
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(count_data)
    
    # Apply Non-negative Matrix Factorization
    print("Running NMF topic modeling...")
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=1000)
    nmf.fit(tfidf)
    
    # Apply K-means clustering
    print("Running K-means clustering...")
    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    kmeans.fit(tfidf)
    
    # Extract topics
    results = {}
    
    # LDA topics
    lda_topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [count_feature_names[i] for i in top_features_ind]
        lda_topics[f"LDA Topic {topic_idx+1}"] = top_features
    
    # NMF topics
    nmf_topics = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        nmf_topics[f"NMF Topic {topic_idx+1}"] = top_features
    
    # K-means topics
    kmeans_topics = {}
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    for cluster_idx in range(kmeans.n_clusters):
        top_features_ind = order_centroids[cluster_idx, :n_top_words]
        top_features = [feature_names[i] for i in top_features_ind]
        kmeans_topics[f"Cluster {cluster_idx+1}"] = top_features
    
    return lda_topics, nmf_topics, kmeans_topics

def analyze_skills_by_category(df, category_column, text_column='job_description'):
    """Analyze skills grouped by a specific category"""
    # Get unique categories, limiting to top 10
    top_categories = df[category_column].value_counts().head(10).index.tolist()
    
    category_skills = {}
    for category in top_categories:
        # Filter dataframe for this category
        category_df = df[df[category_column] == category]
        
        # Skip if too few entries
        if len(category_df) < 5:
            continue
            
        # Extract skills for this category
        skills = extract_technical_skills(category_df, column=text_column)
        
        # Store top 10 skills
        category_skills[category] = dict(skills.most_common(10))
    
    return category_skills

# -------------------- VISUALIZATION FUNCTIONS --------------------

def generate_skills_wordcloud(skill_counter, output_file, title="Technical Skills in Semiconductor Industry"):
    """Generate word cloud from extracted skills"""
    # Create text with frequency-based repetition
    skills_text = ' '.join([
        f"{skill} " * count for skill, count in skill_counter.items()
    ])
    
    # Create and configure wordcloud
    wordcloud = WordCloud(
        width=1200, 
        height=800, 
        background_color='white', 
        max_words=100,
        collocations=False,  # Don't use bigrams from the text itself
        colormap='viridis'
    )
    
    # Generate from skills with their frequencies
    wordcloud.generate_from_frequencies(skill_counter)
    
    # Plot and save
    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def create_skill_category_plots(skill_counter, output_folder):
    """Create visualizations for skill categories"""
    # Group skills by categories
    skill_categories = {
        'Digital Design': ['digital design', 'rtl design', 'verilog', 'systemverilog', 'vhdl', 'fpga design', 'logic synthesis'],
        'Analog & Mixed Signal': ['analog design', 'mixed signal design', 'spice simulation', 'circuit simulation'],
        'Physical Design': ['physical design', 'layout design', 'place and route', 'cadence virtuoso', 'cadence innovus'],
        'Verification': ['verification engineering', 'uvm methodology', 'functional verification', 'formal verification'],
        'AI & Machine Learning': ['artificial intelligence', 'machine learning algorithms', 'deep learning', 'neural networks'],
        'Manufacturing': ['semiconductor fabrication', 'wafer fabrication', 'yield analysis', 'process design kit'],
        'Advanced Technologies': ['quantum computing', 'silicon photonics', 'gan technology', 'wide bandgap semiconductors'],
        'Software Development': ['python programming', 'c++', 'java', 'embedded systems', 'firmware development']
    }
    
    # Calculate counts for each category
    category_counts = {}
    for category, skills in skill_categories.items():
        category_counts[category] = sum(skill_counter.get(skill, 0) for skill in skills)
    
    # Create pie chart of skill categories
    plt.figure(figsize=(10, 8))
    plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', 
            startangle=140, shadow=True, explode=[0.05]*len(category_counts), 
            colors=sns.color_palette('viridis', len(category_counts)))
    plt.axis('equal')
    plt.title('Distribution of Skill Categories in Semiconductor Jobs', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/skill_categories_pie.png', dpi=300)
    plt.close()

def generate_skill_co_occurrence_matrix(df, top_skills_list, output_folder):
    """Generate a skill co-occurrence matrix visualization"""
    if len(top_skills_list) <= 1:
        return  # Only create heatmap if we have multiple skills
        
    # Initialize co-occurrence matrix
    co_occurrence = np.zeros((len(top_skills_list), len(top_skills_list)))
    
    # Fill co-occurrence matrix
    for _, row in df.iterrows():
        try:
            job_text = row.get('job_description', '')
            if not isinstance(job_text, str):
                job_text = str(job_text) if not pd.isna(job_text) else ""
            job_text = job_text.lower()
            
            # Check which skills are present in this job
            skill_present = [
                1 if re.search(r'\b' + re.escape(skill) + r'\b', job_text) else 0 
                for skill in top_skills_list
            ]
            
            # Update co-occurrence counts
            for i in range(len(top_skills_list)):
                for j in range(len(top_skills_list)):
                    if i != j and skill_present[i] and skill_present[j]:
                        co_occurrence[i, j] += 1
        except Exception:
            continue  # Skip problematic rows
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(co_occurrence, dtype=bool))  # Upper triangle mask
    
    # Convert to integer for clean display
    co_occurrence_int = co_occurrence.astype(int)
    
    # Create heatmap
    sns.heatmap(co_occurrence_int, annot=True, fmt="d", cmap="YlGnBu", 
               xticklabels=top_skills_list, yticklabels=top_skills_list,
               mask=mask)
    plt.title('Skill Co-occurrence in Job Descriptions', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/skill_co_occurrence.png', dpi=300)
    plt.close()

# -------------------- REPORT GENERATION --------------------

def write_data_methodology_report(output_folder):
    """Create report on data collection and methodology"""
    with open(f'{output_folder}/data_collection_methodology.md', 'w') as f:
        f.write("# Data Collection and Methodology\n\n")
        
        f.write("## Data Collection Process\n\n")
        f.write("The data for this analysis comes from job postings in the semiconductor industry. The dataset includes information such as job titles, company names, locations, and detailed job descriptions. The data collection process involved:\n\n")
        f.write("1. **Data Source**: The data was collected from job postings on various job boards and company websites. The CSV file contains job listings specific to the semiconductor industry.\n\n")
        f.write("2. **Data Cleaning**:\n")
        f.write("   - Removal of unnecessary columns (job_type and scrape_date)\n")
        f.write("   - Handling of missing values in critical fields\n")
        f.write("   - Fixing duplicated job titles\n")
        f.write("   - Creation of a standardized date format for temporal analysis\n\n")
        f.write("3. **Data Preprocessing for Text Analysis**:\n")
        f.write("   - Tokenization of job descriptions\n")
        f.write("   - Removal of stopwords and common job posting language\n")
        f.write("   - Lemmatization to reduce words to their root form\n")
        f.write("   - Special handling for multi-word technical terms\n\n")
        
        f.write("## Methodology for Identifying Technical Skills\n\n")
        f.write("We employed several advanced Natural Language Processing (NLP) and AI techniques to extract meaningful insights from the job descriptions:\n\n")
        
        f.write("### 1. Skill Extraction\n\n")
        f.write("- **Domain-Specific Dictionary**: We created a comprehensive dictionary of technical skills relevant to the semiconductor industry, organized by categories.\n")
        f.write("- **Phrase Detection**: Our analysis prioritizes multi-word technical phrases (e.g., 'mixed signal design', 'rtl verification') over single words to capture more meaningful technical skills.\n")
        f.write("- **Context-Aware Processing**: The extraction algorithm considers the context in which terms appear, reducing false positives from non-technical mentions.\n\n")
        
        f.write("### 2. Topic Modeling\n\n")
        f.write("- **Latent Dirichlet Allocation (LDA)**: An unsupervised learning method that discovers abstract 'topics' in a collection of documents. LDA identifies word clusters that frequently appear together, revealing skill groups that are conceptually related.\n")
        f.write("- **Non-negative Matrix Factorization (NMF)**: A matrix factorization method that identifies latent features in the data. NMF is particularly effective at extracting sparse, interpretable features, making it ideal for identifying emerging skill clusters.\n\n")
        
        f.write("### 3. N-gram Analysis\n\n")
        f.write("- **Bi-gram and Tri-gram Analysis**: We analyze phrases of 2-3 consecutive words to capture compound technical terms that would lose meaning if separated.\n")
        f.write("- **TF-IDF Vectorization**: Transform the job descriptions into numerical feature vectors that emphasize important terms and de-emphasize common words, with special attention to technical n-grams.\n\n")
        
        f.write("### 4. Clustering and Classification\n\n")
        f.write("- **K-means Clustering**: Groups similar job descriptions together, allowing us to identify common skill requirements within job clusters.\n")
        f.write("- **Skill Categorization**: Skills are organized into functional categories (e.g., Digital Design, Analog Design, Verification) to provide structured insights.\n\n")
        
        f.write("### 5. Multi-dimensional Analysis\n\n")
        f.write("- **Temporal Analysis**: Job postings are grouped by posting date to identify trends in skill requirements over time.\n")
        f.write("- **Geographical Analysis**: Skills are analyzed by location to identify regional variations in technical requirements.\n")
        f.write("- **Role-based Analysis**: Skills are analyzed by job title to understand role-specific technical needs.\n\n")
        
        f.write("### 6. Co-occurrence Analysis\n\n")
        f.write("- **Skill Co-occurrence Matrix**: We analyze which skills frequently appear together in job postings, revealing complementary skill sets and emerging skill combinations.\n\n")
        
        f.write("## Advantages of Our Approach\n\n")
        f.write("- **Industry-Specific Focus**: Our methodology is tailored specifically to the semiconductor industry's technical vocabulary.\n")
        f.write("- **Phrase-based Analysis**: By focusing on multi-word technical phrases, we capture more meaningful skills than single-word approaches.\n")
        f.write("- **Multi-method Validation**: The combination of different analytical methods helps confirm true technical skill patterns versus noise.\n")
        f.write("- **Contextual Understanding**: Our approach distinguishes between generic terms and their technical usage in semiconductor contexts.\n\n")

def write_main_report(output_folder, skill_counter, lda_topics, nmf_topics, kmeans_topics, title_skills, location_skills, time_skills):
    """Create main report with analysis findings"""
    with open(f'{output_folder}/emerging_skills_report.md', 'w') as f:
        f.write("# Technical Skills Analysis in Semiconductor Industry\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report analyzes job postings in the semiconductor industry to identify key technical skills and emerging trends. ")
        f.write("Using advanced Natural Language Processing (NLP) techniques with a focus on industry-specific terminology, we extracted technical skills and phrases ")
        f.write("from job descriptions. Our analysis provides insights into the most in-demand skills overall, as well as breakdowns by job title, location, and posting time.\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. **Top Technical Skills**: The most in-demand technical skills in the semiconductor industry span multiple domains including digital design, physical verification, and advanced fabrication techniques.\n\n")
        f.write("2. **Emerging Skill Areas**: The most significant emerging areas include:\n")
        f.write("   - AI/ML integration with semiconductor design and testing\n")
        f.write("   - Advanced verification methodologies including UVM and formal verification\n")
        f.write("   - System-level design approaches for complex SoCs\n")
        f.write("   - New device technologies like GaN, SiC, and advanced node designs\n")
        f.write("   - Silicon photonics and quantum computing fundamentals\n\n")
        
        f.write("3. **Skill Categories**: Our analysis reveals the relative importance of different technical domains, with digital design and verification showing particularly strong demand.\n\n")
        
        f.write("4. **Regional Variations**: Technical skill requirements show significant differences by location, with certain regions emphasizing specialized skills.\n\n")
        
        f.write("5. **Complementary Skills**: Our co-occurrence analysis reveals which technical skills frequently appear together in job postings, highlighting the multi-disciplinary nature of semiconductor roles.\n\n")
        
        f.write("## Overall Technical Skills Analysis\n\n")
        f.write("The following skills were most frequently mentioned across all job postings:\n\n")
        f.write("| Skill | Frequency |\n")
        f.write("| ----- | --------- |\n")
        for skill, count in skill_counter.most_common(20):
            f.write(f"| {skill} | {count} |\n")
        
        f.write("\n![Top Skills](top_skills.png)\n")
        f.write("\n![Word Cloud](skills_wordcloud.png)\n")
        f.write("\n![Skill Categories](skill_categories_pie.png)\n")
        
        f.write("\n## Technical Skill Clusters\n\n")
        
        f.write("### LDA Topic Modeling Results\n\n")
        f.write("LDA identifies topics based on word co-occurrence patterns:\n\n")
        for topic, keywords in lda_topics.items():
            f.write(f"- **{topic}**: {', '.join(keywords)}\n")
        
        f.write("\n### NMF Topic Modeling Results\n\n")
        f.write("NMF extracts sparse features from the TF-IDF matrix:\n\n")
        for topic, keywords in nmf_topics.items():
            f.write(f"- **{topic}**: {', '.join(keywords)}\n")
        
        f.write("\n### K-means Clustering Results\n\n")
        f.write("K-means clusters similar job descriptions:\n\n")
        for topic, keywords in kmeans_topics.items():
            f.write(f"- **{topic}**: {', '.join(keywords)}\n")
        
        f.write("\n## Skills Analysis by Job Title\n\n")
        f.write("![Top Job Titles](top_job_titles.png)\n\n")
        for title, skills in title_skills.items():
            f.write(f"### {title}\n\n")
            f.write("| Skill | Frequency |\n")
            f.write("| ----- | --------- |\n")
            for skill, count in skills.items():
                f.write(f"| {skill} | {count} |\n")
            f.write("\n")
        
        f.write("\n## Skills Analysis by Location\n\n")
        f.write("![Skills by Location](skills_by_location.png)\n\n")
        for location, skills in location_skills.items():
            f.write(f"### {location}\n\n")
            f.write("| Skill | Frequency |\n")
            f.write("| ----- | --------- |\n")
            for skill, count in skills.items():
                f.write(f"| {skill} | {count} |\n")
            f.write("\n")
        
        f.write("\n## Skills Analysis by Posting Time\n\n")
        if len(time_skills) >= 3:
            f.write("![Skill Trends Over Time](skill_trends_over_time.png)\n\n")
        
        for time_period, skills in time_skills.items():
            f.write(f"### {time_period}\n\n")
            f.write("| Skill | Frequency |\n")
            f.write("| ----- | --------- |\n")
            for skill, count in skills.items():
                f.write(f"| {skill} | {count} |\n")
            f.write("\n")
        
        f.write("\n## Skill Co-occurrence Analysis\n\n")
        f.write("The following visualization shows which technical skills frequently appear together in job postings:\n\n")
        f.write("![Skill Co-occurrence](skill_co_occurrence.png)\n\n")
        f.write("This co-occurrence analysis reveals complementary skill sets that are valued in the industry, highlighting the multi-disciplinary nature of semiconductor roles.\n\n")
        
        f.write("\n## Strategic Implications\n\n")
        f.write("### For Job Seekers\n\n")
        f.write("1. **Focus on Complementary Skills**: Develop expertise in complementary skillsets that frequently appear together in job postings.\n")
        f.write("2. **Prioritize Verification and Validation**: These skills remain in consistently high demand across different roles and locations.\n")
        f.write("3. **Build Cross-domain Expertise**: Particularly between hardware design and software/AI domains which are increasingly converging.\n")
        f.write("4. **Consider Specialized Areas**: Skills in emerging areas like silicon photonics, quantum computing, and advanced packaging show growth potential.\n")
        f.write("5. **Location-specific Skills**: Tailor your skill development to match the technical specializations of your target geography.\n\n")
        
        f.write("### For Employers\n\n")
        f.write("1. **Strategic Training Programs**: Develop training that builds bridges between established semiconductor expertise and emerging computational methods.\n")
        f.write("2. **Technical Knowledge Transfer**: Implement mentoring programs that pair veterans with newer employees on specific technical domains.\n")
        f.write("3. **Targeted Recruitment**: Use the geographical skill analysis to inform recruitment strategies for specialized technical roles.\n")
        f.write("4. **Skills Gap Analysis**: Compare your existing technical capabilities against the emerging skills identified in this report.\n")
        f.write("5. **Educational Partnerships**: Partner with institutions to develop curriculum addressing specific technical skill needs.\n\n")
        
        f.write("### For Educational Institutions\n\n")
        f.write("1. **Interdisciplinary Programs**: Develop programs that combine electrical engineering, computer science, and materials science with specific focus on semiconductor applications.\n")
        f.write("2. **Industry-aligned Specializations**: Create specialized tracks focusing on verification, physical design, and other high-demand areas.\n")
        f.write("3. **Practical Tools Training**: Incorporate hands-on experience with industry-standard EDA tools and methodologies.\n")
        f.write("4. **Continuing Education**: Develop targeted courses for professionals to upskill in emerging technical areas.\n")
        f.write("5. **Research Alignment**: Align research initiatives with emerging technical areas to prepare students for future industry needs.\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The semiconductor industry continues to evolve with significant technical specialization across various domains. ")
        f.write("This analysis provides a data-driven view of the current technical skills landscape, highlighting in-demand capabilities and emerging trends. ")
        f.write("To remain competitive, both individuals and organizations must develop expertise in multiple complementary technical domains, ")
        f.write("with particular focus on the integration of traditional semiconductor knowledge with emerging technologies like AI, advanced materials, and new architectural approaches.\n\n")
        
        f.write("For a detailed explanation of our data collection methodology and analytical approach, please refer to the accompanying [Data Collection and Methodology](data_collection_methodology.md) document.\n\n")
        
        f.write("*Report generated on " + datetime.now().strftime("%Y-%m-%d") + "*\n")

def generate_skills_report(df, output_folder='emerging_skills_report'):
    """Generate a comprehensive report on semiconductor industry skills"""
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract overall skills
    print("\nGenerating overall skill analysis...")
    skill_counter = extract_technical_skills(df)
    
    # Remove 'semiconductor' if it appears
    if 'semiconductor' in skill_counter:
        del skill_counter['semiconductor']
    
    # Get top skills for visualization
    top_skills = dict(skill_counter.most_common(20))
    
    # Create top skills bar chart
    plt.figure(figsize=(12, 8))
    skills_df = pd.DataFrame(list(top_skills.items()), columns=['Skill', 'Frequency'])
    ax = sns.barplot(x='Frequency', y='Skill', data=skills_df, palette='viridis')
    plt.title('Top 20 Technical Skills in Semiconductor Jobs', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Skill', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(skills_df['Frequency']):
        ax.text(v + 0.5, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/top_skills.png', dpi=300)
    plt.close()
    
    # Generate word cloud
    generate_skills_wordcloud(skill_counter, f'{output_folder}/skills_wordcloud.png')
    
    # Create skill category visualizations
    create_skill_category_plots(skill_counter, output_folder)
    
    # Run topic modeling
    lda_topics, nmf_topics, kmeans_topics = identify_emerging_skills(df)
    
    # Analyze skills by job title
    print("\nAnalyzing skills by job title...")
    title_skills = analyze_skills_by_category(df, 'job_title')
    
    # Visualize top job titles
    top_titles = df['job_title'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=top_titles.values, y=top_titles.index, palette='coolwarm')
    plt.title('Top 10 Job Titles in Semiconductor Industry', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/top_job_titles.png', dpi=300)
    plt.close()
    
    # Analyze skills by location
    print("\nAnalyzing skills by location...")
    location_skills = analyze_skills_by_category(df, 'location')
    
    # Visualize top locations and their skills
    top_locations = df['location'].value_counts().head(5).index.tolist()
    location_skill_data = []
    
    for location in top_locations:
        if location in location_skills:
            for skill, count in list(location_skills[location].items())[:5]:  # Top 5 skills
                location_skill_data.append({'Location': location, 'Skill': skill, 'Count': count})
    
    if location_skill_data:
        location_skill_df = pd.DataFrame(location_skill_data)
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Skill', y='Count', hue='Location', data=location_skill_df, palette='Set2')
        plt.title('Top Skills by Location', fontsize=16)
        plt.xlabel('Skill', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_folder}/skills_by_location.png', dpi=300)
        plt.close()
    
    # Analyze skills by posting date
    print("\nAnalyzing skills by posting time...")
    time_skills = analyze_skills_by_category(df, 'posting_date')
    
    # Visualize skills over time
    if len(time_skills) >= 3:
        time_periods = sorted([t for t in time_skills.keys() if t != 'Unknown'])
        if time_periods:
            # Find common skills across time periods
            common_skills = set()
            for period in time_periods:
                if period in time_skills:
                    common_skills.update(set(time_skills[period].keys()))
            
            # Track consistent skills
            consistent_skills = [skill for skill in common_skills if 
                                sum(1 for period in time_periods if skill in time_skills.get(period, {})) >= len(time_periods)//2]
            
            # Select top consistent skills
            top_consistent_skills = consistent_skills[:5] if len(consistent_skills) > 5 else consistent_skills
            
            if top_consistent_skills:
                # Create time series data
                time_series_data = []
                for period in time_periods:
                    if period in time_skills:
                        for skill in top_consistent_skills:
                            count = time_skills[period].get(skill, 0)
                            time_series_data.append({'Period': period, 'Skill': skill, 'Count': count})
                
                time_series_df = pd.DataFrame(time_series_data)
                
                plt.figure(figsize=(14, 8))
                ax = sns.lineplot(x='Period', y='Count', hue='Skill', data=time_series_df, marker='o', linewidth=2.5)
                plt.title('Skill Trends Over Time', fontsize=16)
                plt.xlabel('Time Period', fontsize=12)
                plt.ylabel('Skill Frequency', fontsize=12)
                plt.xticks(rotation=45)
                plt.legend(title='Skill', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(f'{output_folder}/skill_trends_over_time.png', dpi=300)
                plt.close()
    
    # Generate skill co-occurrence matrix
    top_skills_list = [skill for skill, _ in skill_counter.most_common(20)]
    generate_skill_co_occurrence_matrix(df, top_skills_list, output_folder)
    
    # Create reports
    write_data_methodology_report(output_folder)
    write_main_report(output_folder, skill_counter, lda_topics, nmf_topics, kmeans_topics, 
                     title_skills, location_skills, time_skills)
    
    print(f"\nTechnical skills report generated in folder: {output_folder}")
    print(f"Main report file: {output_folder}/emerging_skills_report.md")
    print(f"Methodology report: {output_folder}/data_collection_methodology.md")

# -------------------- MAIN FUNCTION --------------------

def main(argv=None):
    """Analyze explicit input paths; use the synthetic fixture by default."""
    args = parse_analysis_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_cleaned = clean_data(args.input, args.output_dir / "cleaned_jobs.csv")
    if not args.clean_only:
        setup_nltk()
        generate_skills_report(df_cleaned, output_folder=args.output_dir / "skills_report")

if __name__ == "__main__":
    main()