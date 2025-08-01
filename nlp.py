"""
AI Interview Generator -  AI-Powered Resume Analyzer Pro
Copyright (c) 2025 Isha Verma. All rights reserved.
Version: 1.0

This software is the intellectual property of Isha Verma.
Unauthorized copying, distribution, or modification is prohibited.
"""

import random
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string
from typing import List, Dict, Tuple, Set
import spacy
from textblob import TextBlob

class AdvancedNLPProcessor:
    """
    Advanced NLP processing utilities for resume and job description analysis
    """
    
    def __init__(self):
        self.download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Extended skill databases
        self.skill_keywords = {
            'programming_languages': {
                'python': ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
                'javascript': ['javascript', 'js', 'node.js', 'nodejs', 'react', 'angular', 'vue'],
                'java': ['java', 'spring', 'hibernate', 'maven', 'gradle'],
                'c++': ['c++', 'cpp', 'c plus plus'],
                'c#': ['c#', 'csharp', 'c sharp', '.net', 'dotnet'],
                'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'oracle'],
                'r': ['r programming', 'rstudio', 'tidyverse', 'ggplot2'],
                'go': ['golang', 'go lang'],
                'rust': ['rust lang', 'cargo'],
                'scala': ['scala', 'akka', 'play framework'],
                'php': ['php', 'laravel', 'symfony', 'codeigniter'],
                'ruby': ['ruby', 'rails', 'ruby on rails'],
                'swift': ['swift', 'ios development', 'xcode'],
                'kotlin': ['kotlin', 'android development']
            },
            
            'web_technologies': {
                'frontend': ['html', 'css', 'scss', 'sass', 'less', 'bootstrap', 'tailwind'],
                'react': ['react', 'reactjs', 'redux', 'next.js', 'gatsby'],
                'angular': ['angular', 'angularjs', 'typescript', 'rxjs'],
                'vue': ['vue.js', 'vuejs', 'nuxt.js', 'vuex'],
                'backend': ['node.js', 'express', 'koa', 'fastify'],
                'apis': ['rest api', 'restful', 'graphql', 'soap', 'json', 'xml']
            },
            
            'data_science': {
                'libraries': ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly'],
                'ml_frameworks': ['scikit-learn', 'tensorflow', 'pytorch', 'keras', 'xgboost'],
                'deep_learning': ['neural networks', 'cnn', 'rnn', 'lstm', 'transformer'],
                'statistics': ['statistics', 'probability', 'hypothesis testing', 'regression'],
                'tools': ['jupyter', 'anaconda', 'spyder', 'r studio', 'tableau', 'power bi']
            },
            
            'cloud_devops': {
                'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'rds', 'cloudfront'],
                'azure': ['azure', 'microsoft azure', 'azure functions', 'cosmos db'],
                'gcp': ['google cloud', 'gcp', 'compute engine', 'app engine', 'bigquery'],
                'containers': ['docker', 'kubernetes', 'k8s', 'container', 'containerization'],
                'ci_cd': ['jenkins', 'gitlab ci', 'github actions', 'travis ci', 'circleci'],
                'infrastructure': ['terraform', 'ansible', 'chef', 'puppet', 'cloudformation']
            },
            
            'databases': {
                'relational': ['mysql', 'postgresql', 'sqlite', 'oracle', 'sql server'],
                'nosql': ['mongodb', 'cassandra', 'couchdb', 'dynamodb'],
                'cache': ['redis', 'memcached', 'elasticsearch', 'solr'],
                'big_data': ['hadoop', 'spark', 'kafka', 'hive', 'pig']
            },
            
            'soft_skills': {
                'leadership': ['leadership', 'team lead', 'management', 'mentoring', 'coaching'],
                'communication': ['communication', 'presentation', 'public speaking', 'writing'],
                'collaboration': ['teamwork', 'collaboration', 'cross-functional', 'agile', 'scrum'],
                'problem_solving': ['problem solving', 'analytical', 'critical thinking', 'debugging'],
                'project_management': ['project management', 'pmp', 'agile', 'waterfall', 'kanban']
            }
        }
        
        # Industry-specific keywords
        self.industry_keywords = {
            'fintech': ['fintech', 'blockchain', 'cryptocurrency', 'trading', 'risk management'],
            'healthcare': ['healthcare', 'medical', 'hipaa', 'clinical', 'pharmaceutical'],
            'ecommerce': ['ecommerce', 'retail', 'payment', 'inventory', 'supply chain'],
            'gaming': ['gaming', 'unity', 'unreal', 'game development', 'graphics'],
            'ai_ml': ['artificial intelligence', 'machine learning', 'deep learning', 'nlp', 'computer vision']
        }
        
        # Experience level indicators
        self.experience_indicators = {
            'junior': ['junior', 'entry level', 'graduate', 'intern', '0-2 years'],
            'mid': ['mid level', 'intermediate', '3-5 years', 'experienced'],
            'senior': ['senior', 'lead', 'principal', '5+ years', 'expert', 'architect']
        }
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-\+\#\.\,]', ' ', text)
        
        # Handle specific patterns
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', text)  # Year ranges
        text = re.sub(r'(\w+)\s*\+\s*(\w+)', r'\1+\2', text)  # C++, etc.
        
        return text.strip()
    
    def extract_skills_advanced(self, text: str) -> Dict[str, List[str]]:
        """Extract skills using advanced pattern matching"""
        text_clean = self.clean_text(text)
        found_skills = {}
        
        for category, subcategories in self.skill_keywords.items():
            found_skills[category] = {}
            
            for subcategory, skills in subcategories.items():
                found_skills[category][subcategory] = []
                
                for skill in skills:
                    # Use word boundaries for exact matching
                    pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                    if re.search(pattern, text_clean):
                        found_skills[category][subcategory].append(skill)
        
        # Flatten and return non-empty categories
        result = {}
        for category, subcategories in found_skills.items():
            category_skills = []
            for subcategory, skills in subcategories.items():
                category_skills.extend(skills)
            if category_skills:
                result[category] = list(set(category_skills))
        
        return result
    
    def extract_experience_level(self, text: str) -> str:
        """Extract experience level from text"""
        text_clean = self.clean_text(text)
        
        for level, indicators in self.experience_indicators.items():
            for indicator in indicators:
                if indicator in text_clean:
                    return level
        
        # Try to extract years of experience
        years_pattern = r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience'
        matches = re.findall(years_pattern, text_clean)
        
        if matches:
            years = int(matches[0])
            if years <= 2:
                return 'junior'
            elif years <= 5:
                return 'mid'
            else:
                return 'senior'
        
        return 'unknown'
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        text_clean = self.clean_text(text)
        
        education_patterns = [
            r'\b(phd|ph\.d|doctorate|doctoral)\b',
            r'\b(master\'?s?|m\.s|m\.a|mba|m\.tech|m\.eng)\b',
            r'\b(bachelor\'?s?|b\.s|b\.a|b\.tech|b\.eng|be|btech)\b',
            r'\b(associate|diploma|certificate)\b'
        ]
        
        found_education = []
        for pattern in education_patterns:
            matches = re.findall(pattern, text_clean)
            found_education.extend(matches)
        
        return list(set(found_education))
    
    def calculate_keyword_density(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword density in text"""
        if not keywords:
            return 0.0
        
        text_clean = self.clean_text(text)
        words = word_tokenize(text_clean)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        keyword_count = 0
        for keyword in keywords:
            keyword_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            keyword_count += len(re.findall(keyword_pattern, text_clean))
        
        return (keyword_count / total_words) * 100
    
    def extract_achievements(self, text: str) -> List[str]:
        """Extract achievement indicators from text"""
        achievement_patterns = [
            r'(?:increased|improved|reduced|achieved|delivered|led|managed|developed|created|built|designed|implemented)\s+[^.]*?(?:\d+%|\$\d+|[\d,]+\s+(?:users|customers|projects|people))',
            r'(?:awarded|recognized|certified|promoted|selected)',
            r'(?:published|presented|spoke at)',
            r'(?:managed\s+(?:team|budget|project)s?\s+of\s+[\d,]+)',
        ]
        
        achievements = []
        for pattern in achievement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            achievements.extend(matches)
        
        return achievements[:10]  # Limit to top 10 achievements
    
    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """Analyze text complexity metrics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Filter out punctuation
        words = [word for word in words if word not in string.punctuation]
        
        if not sentences or not words:
            return {'readability': 0, 'avg_sentence_length': 0, 'complexity_score': 0}
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Syllable count (simplified)
        def count_syllables(word):
            return max(1, len(re.findall(r'[aeiouy]', word.lower())))
        
        total_syllables = sum(count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Flesch Reading Ease (simplified)
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        
        # Complexity score based on unique words and technical terms
        unique_words = len(set(words))
        vocabulary_richness = unique_words / len(words)
        
        complexity_score = min(100, max(0, (100 - flesch_score) + vocabulary_richness * 50))
        
        return {
            'readability': max(0, min(100, flesch_score)),
            'avg_sentence_length': avg_sentence_length,
            'complexity_score': complexity_score,
            'vocabulary_richness': vocabulary_richness
        }
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume text"""
        contact_info = {}
        
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone'] = '-'.join(phones[0])
        
        # LinkedIn
        linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/pub/)([A-Za-z0-9\-]+)'
        linkedin_matches = re.findall(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_matches:
            contact_info['linkedin'] = linkedin_matches[0]
        
        # GitHub
        github_pattern = r'(?:github\.com/)([A-Za-z0-9\-]+)'
        github_matches = re.findall(github_pattern, text, re.IGNORECASE)
        if github_matches:
            contact_info['github'] = github_matches[0]
        
        return contact_info
    
    def semantic_similarity_advanced(self, text1: str, text2: str) -> float:
        """Calculate advanced semantic similarity"""
        # Clean texts
        text1_clean = self.clean_text(text1)
        text2_clean = self.clean_text(text2)
        
        # Use TextBlob for basic semantic analysis
        blob1 = TextBlob(text1_clean)
        blob2 = TextBlob(text2_clean)
        
        # Extract noun phrases (key concepts)
        phrases1 = set(phrase.lower() for phrase in blob1.noun_phrases)
        phrases2 = set(phrase.lower() for phrase in blob2.noun_phrases)
        
        # Calculate phrase overlap
        common_phrases = phrases1.intersection(phrases2)
        total_phrases = phrases1.union(phrases2)
        
        phrase_similarity = len(common_phrases) / len(total_phrases) if total_phrases else 0
        
        # Word-level similarity
        words1 = set(word.lower() for word in blob1.words if word.lower() not in self.stop_words)
        words2 = set(word.lower() for word in blob2.words if word.lower() not in self.stop_words)
        
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        
        word_similarity = len(common_words) / len(total_words) if total_words else 0
        
        # Combined similarity (weighted)
        combined_similarity = (phrase_similarity * 0.6 + word_similarity * 0.4)
        
        return combined_similarity
    
    def extract_job_requirements(self, jd_text: str) -> Dict[str, any]:
        """Extract structured requirements from job description"""
        text_clean = self.clean_text(jd_text)
        
        requirements = {
            'must_have_skills': [],
            'nice_to_have_skills': [],
            'experience_required': self.extract_experience_level(jd_text),
            'education_required': self.extract_education(jd_text),
            'industry': self._detect_industry(text_clean),
            'role_type': self._detect_role_type(text_clean),
            'remote_friendly': self._detect_remote_work(text_clean)
        }
        
        # Extract must-have vs nice-to-have skills
        must_have_patterns = [
            r'(?:required|must have|essential|mandatory)[\s\w]*?:(.*?)(?:\n|\.)',
            r'(?:minimum|required)\s+(?:qualifications|requirements|skills)[\s\w]*?:(.*?)(?:\n|\.)',
        ]
        
        nice_to_have_patterns = [
            r'(?:preferred|nice to have|bonus|plus|desired)[\s\w]*?:(.*?)(?:\n|\.)',
            r'(?:additional|preferred)\s+(?:qualifications|requirements|skills)[\s\w]*?:(.*?)(?:\n|\.)',
        ]
        
        for pattern in must_have_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                requirements['must_have_skills'].extend(skills)
        
        for pattern in nice_to_have_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                requirements['nice_to_have_skills'].extend(skills)
        
        # Remove duplicates
        requirements['must_have_skills'] = list(set(requirements['must_have_skills']))
        requirements['nice_to_have_skills'] = list(set(requirements['nice_to_have_skills']))
        
        return requirements
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Helper method to extract skills from a text snippet"""
        skills = []
        all_skills = self.extract_skills_advanced(text)
        
        for category, skill_list in all_skills.items():
            skills.extend(skill_list)
        
        return skills
    
    def _detect_industry(self, text: str) -> str:
        """Detect industry from job description"""
        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return industry
        return 'general'
    
    def _detect_role_type(self, text: str) -> str:
        """Detect role type from job description"""
        role_indicators = {
            'software_engineer': ['software engineer', 'developer', 'programmer', 'coding'],
            'data_scientist': ['data scientist', 'data analyst', 'machine learning', 'analytics'],
            'product_manager': ['product manager', 'product owner', 'pm', 'product strategy'],
            'designer': ['designer', 'ui', 'ux', 'user experience', 'user interface'],
            'marketing': ['marketing', 'digital marketing', 'content marketing', 'seo'],
            'sales': ['sales', 'business development', 'account manager', 'sales rep']
        }
        
        for role, indicators in role_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    return role
        
        return 'general'
    
    def _detect_remote_work(self, text: str) -> bool:
        """Detect if role supports remote work"""
        remote_indicators = ['remote', 'work from home', 'wfh', 'distributed', 'anywhere']
        
        for indicator in remote_indicators:
            if indicator in text:
                return True
        
        return False