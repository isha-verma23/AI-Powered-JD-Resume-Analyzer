"""
AI Interview Generator -  AI-Powered Resume Analyzer Pro
Copyright (c) 2025 Isha Verma. All rights reserved.
Version: 1.0

This software is the intellectual property of Isha Verma.
Unauthorized copying, distribution, or modification is prohibited.
"""

import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field 
from datetime import datetime
import json

@dataclass
class ScoreBreakdown:
    overall_score: float
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    semantic_similarity_score: float
    keyword_density_score: float
    content_quality: float
    role_fit_score: float
    ats_compatibility: float
    confidence_interval: Tuple[float, float]
    detailed_breakdown: Dict[str, Any]
    ats_details: Dict[str, Any] = field(default_factory=dict)

class AdvancedScoringEngine:
    """
    Advanced scoring engine using multiple ML techniques for resume-JD matching
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True,
            token_pattern=r'\b[A-Za-z][A-Za-z0-9+#]*\b'
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.scaler = StandardScaler()
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        
        # Weighted importance for different sections
        self.section_weights = {
            'skills': 0.35,
            'experience': 0.30,
            'education': 0.15,
            'summary': 0.10,
            'projects': 0.07,
            'certifications': 0.03
        }
        
        # Experience level mapping
        self.experience_levels = {
            'entry': {'min_years': 0, 'max_years': 2, 'keywords': ['entry', 'junior', 'graduate', 'intern']},
            'mid': {'min_years': 2, 'max_years': 5, 'keywords': ['mid', 'intermediate', 'experienced']},
            'senior': {'min_years': 5, 'max_years': 10, 'keywords': ['senior', 'lead', 'principal']},
            'executive': {'min_years': 10, 'max_years': float('inf'), 'keywords': ['director', 'manager', 'vp', 'executive']}
        }
        
        # Industry-specific skill importance
        self.industry_skill_weights = {
            'tech': {
                'programming_languages': 0.4,
                'web_technologies': 0.3,
                'databases': 0.2,
                'cloud_devops': 0.1
            },
            'data_science': {
                'data_science': 0.5,
                'programming_languages': 0.3,
                'databases': 0.2
            },
            'marketing': {
                'soft_skills': 0.4,
                'tools': 0.3,
                'web_technologies': 0.3
            }
        }
    
    def calculate_comprehensive_match_score(
        self, 
        resume_data: Dict, 
        jd_data: Dict,
        job_requirements: Dict = None
    ) -> ScoreBreakdown:
        """
        Calculate comprehensive match score using multiple algorithms
        """
        
        # Extract texts
        resume_text = resume_data.get('raw_text', '')
        jd_text = jd_data.get('raw_text', jd_data.get('text', ''))
        
        # Initialize score components
        scores = {}
        
        # 1. Skill Matching Score
        scores['skill_match_score'] = self._calculate_skill_match_score(
            resume_data, jd_data, job_requirements
        )
        
        # 2. Experience Matching Score
        scores['experience_match_score'] = self._calculate_experience_match_score(
            resume_data, jd_data, job_requirements
        )
        
        # 3. Education Matching Score
        scores['education_match_score'] = self._calculate_education_match_score(
            resume_data, jd_data, job_requirements
        )
        
        # 4. Semantic Similarity Score
        scores['semantic_similarity_score'] = self._calculate_semantic_similarity_score(
            resume_text, jd_text
        )
        
        # 5. Keyword Density Score
        scores['keyword_density'] = self._calculate_keyword_density_score(
            resume_text, jd_text
        )
        
        # 6. Role Fit Score
        scores['role_fit'] = self._calculate_role_fit_score(
            resume_data, jd_data, job_requirements
        )
        
        # 7. Content Quality Score
        scores['content_quality'] = self._calculate_content_quality_score(resume_data)

        # 8. ATS Compatibility Score
        ats = self._calculate_ats_compatibility_score(resume_data)
        scores['ats_compatibility'] = ats['overall_score']   # float ONLY
        
        # Calculate weighted overall score
        weights = {
            'skill_match_score': 0.25,
            'experience_match_score': 0.20,
            'education_match_score': 0.10,
            'semantic_similarity_score': 0.15,
            'keyword_density': 0.10,
            'role_fit': 0.15,
            'content_quality': 0.05,
            'ats_compatibility'  : 0.05,
            }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        # Calculate confidence interval
        score_variance = np.var(list(scores.values()))
        confidence_interval = self._calculate_confidence_interval(overall_score, score_variance)
        
        return ScoreBreakdown(
            overall_score=round(overall_score, 2),
            skill_match_score=round(scores['skill_match_score'], 2),
            experience_match_score=round(scores['experience_match_score'], 2),
            education_match_score=round(scores['education_match_score'], 2),
            semantic_similarity_score=round(scores['semantic_similarity_score'], 2),
            keyword_density_score=round(scores['keyword_density'], 2),
            content_quality=round(scores['content_quality'], 2),
            ats_compatibility=round(scores['ats_compatibility'], 2),
            role_fit_score=round(scores['role_fit'], 2),
            confidence_interval=confidence_interval,
            detailed_breakdown=scores,
            ats_details= ats
        )
    
    def _calculate_skill_match_score(
        self, 
        resume_data: Dict, 
        jd_data: Dict, 
        job_requirements: Dict = None
    ) -> float:
        """Calculate skill matching score with weighted importance"""
        
        resume_skills = self._extract_skills_from_data(resume_data)
        jd_skills = self._extract_skills_from_data(jd_data)
        
        if not jd_skills:
            return 50.0  # Neutral score if no skills detected
        
        total_score = 0.0
        total_weight = 0.0
        
        # Calculate score for each skill category
        for category in jd_skills:
            if category in resume_skills:
                resume_category_skills = set(resume_skills[category])
                jd_category_skills = set(jd_skills[category])
                
                # Calculate Jaccard similarity
                intersection = resume_category_skills.intersection(jd_category_skills)
                union = resume_category_skills.union(jd_category_skills)
                
                if union:
                    category_score = len(intersection) / len(union) * 100
                else:
                    category_score = 0.0
                
                # Apply category weight
                weight = self._get_category_weight(category, job_requirements)
                total_score += category_score * weight
                total_weight += weight
            else:
                # Penalty for missing category
                weight = self._get_category_weight(category, job_requirements)
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_experience_match_score(
        self, 
        resume_data: Dict, 
        jd_data: Dict, 
        job_requirements: Dict = None
    ) -> float:
        """Calculate experience matching score"""
        
        # Extract years of experience from resume
        resume_years = self._extract_years_of_experience(resume_data.get('raw_text', ''))
        
        # Extract required experience from JD
        required_years = self._extract_required_experience(jd_data.get('raw_text', jd_data.get('text', '')))
        
        if required_years is None:
            return 75.0  # Neutral score if requirements unclear
        
        # Calculate experience gap
        if resume_years >= required_years:
            # Bonus for overqualification, but with diminishing returns
            excess = resume_years - required_years
            bonus = min(25, excess * 5)  # Max 25 point bonus
            return min(100, 100 + bonus)
        else:
            # Penalty for under-qualification
            deficit = required_years - resume_years
            penalty = deficit * 15  # 15 points per missing year
            return max(0, 100 - penalty)
    
    def _calculate_education_match_score(
        self, 
        resume_data: Dict, 
        jd_data: Dict, 
        job_requirements: Dict = None
    ) -> float:
        """Calculate education matching score"""
        
        resume_education = self._extract_education_level(resume_data.get('raw_text', ''))
        required_education = self._extract_required_education(jd_data.get('raw_text', jd_data.get('text', '')))
        
        education_hierarchy = {
            'high_school': 1,
            'associate': 2,
            'bachelor': 3,
            'master': 4,
            'phd': 5
        }
        
        resume_level = education_hierarchy.get(resume_education, 0)
        required_level = education_hierarchy.get(required_education, 2)  # Default to bachelor's
        
        if resume_level >= required_level:
            return 100.0
        elif resume_level == required_level - 1:
            return 75.0  # Close enough
        else:
            return max(0, 100 - (required_level - resume_level) * 25)
    
    def _calculate_semantic_similarity_score(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity using advanced NLP techniques"""
        
        if not resume_text or not jd_text:
            return 0.0
        
        try:
            # TF-IDF similarity
            texts = [resume_text, jd_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Topic modeling similarity
            count_matrix = self.count_vectorizer.fit_transform(texts)
            lda_matrix = self.lda_model.fit_transform(count_matrix)
            lda_similarity = cosine_similarity(lda_matrix[0:1], lda_matrix[1:2])[0][0]
            
            # Combine similarities
            combined_similarity = (tfidf_similarity * 0.7 + lda_similarity * 0.3) * 100
            
            return max(0, min(100, combined_similarity))
            
        except Exception:
            return 50.0  # Fallback score
    
    def _calculate_keyword_density_score(self, resume_text: str, jd_text: str) -> float:
        """Calculate keyword density score"""
        
        if not resume_text or not jd_text:
            return 0.0
        
        # Extract important keywords from JD
        jd_keywords = self._extract_important_keywords(jd_text)
        
        if not jd_keywords:
            return 50.0
        
        # Calculate density in resume
        resume_lower = resume_text.lower()
        total_keywords = len(jd_keywords)
        found_keywords = 0
        
        for keyword in jd_keywords:
            if keyword.lower() in resume_lower:
                found_keywords += 1
        
        density_score = (found_keywords / total_keywords) * 100
        return min(100, density_score)
    
    def _calculate_role_fit_score(
        self, 
        resume_data: Dict, 
        jd_data: Dict, 
        job_requirements: Dict = None
    ) -> float:
        """Calculate role fit score based on job-specific requirements"""
        
        if not job_requirements:
            return 70.0  # Neutral score
        
        role_type = job_requirements.get('role_type', 'general')
        industry = job_requirements.get('industry', 'general')
        
        # Get role-specific scoring weights
        role_weights = self._get_role_specific_weights(role_type, industry)
        
        # Calculate weighted score based on role requirements
        total_score = 0.0
        total_weight = 0.0
        
        for component, weight in role_weights.items():
            component_score = self._calculate_component_score(
                resume_data, jd_data, component, job_requirements
            )
            total_score += component_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 70.0
    
    def _calculate_content_quality_score(self, resume_data: Dict) -> float:
        """Calculate content quality score"""
        
        resume_text = resume_data.get('raw_text', '')
        sections = resume_data.get('sections', {})
        
        if not resume_text:
            return 0.0
        
        quality_factors = []
        
        # Length appropriateness (1-2 pages ideal)
        word_count = len(resume_text.split())
        if 300 <= word_count <= 800:
            quality_factors.append(100)
        elif 200 <= word_count <= 1200:
            quality_factors.append(80)
        else:
            quality_factors.append(60)
        
        # Section completeness
        essential_sections = ['experience', 'education', 'skills']
        present_sections = sum(1 for section in essential_sections if section in sections)
        section_score = (present_sections / len(essential_sections)) * 100
        quality_factors.append(section_score)
        
        # Writing quality (grammar, sentiment)
        blob = TextBlob(resume_text)
        sentiment_score = max(0, (blob.sentiment.polarity + 1) * 50)  # Convert to 0-100
        quality_factors.append(sentiment_score)
        
        # Contact information completeness
        contact_info = resume_data.get('contact_info', {})
        contact_score = min(100, len(contact_info) * 25)  # 25 points per contact field
        quality_factors.append(contact_score)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _calculate_ats_compatibility_score(self, resume_data: Dict, job_keywords: List[str] = None) -> Dict:

        ats_factors = {
            'file_format': 0,  # Assume PDF is good
            'section_headers': 0,
            'contact_info': 0,
            'keyword_usage': 0,
            'formatting': 0,
            'length': 0,
            'bullet_points': 0
        }
        resume_text = resume_data.get('raw_text', '')
        sections = resume_data.get('sections', {})
        contact_info = resume_data.get('contact_info', {})

        if not resume_text:
            return {
                'overall_score': 0.0,
                'detailed_scores': ats_factors,
                'recommendations': []
            }

        ats_factors['file_format'] = 100

        # 1. Presence of standard section headers
        standard_sections = ['experience', 'education', 'skills', 'summary', 'projects', 'certifications']
        section_coverage = sum(1 for sec in standard_sections if sec in sections)
        ats_factors['section_headers'] = (section_coverage / len(standard_sections)) * 100

        # Contact information
        essential_contact = ['email', 'phone']
        present_contact = sum(1 for field in essential_contact if field in contact_info)
        ats_factors['contact_info'] = (present_contact / len(essential_contact)) * 100

    # 2. Keyword matching (if job_keywords are provided)
        if job_keywords:
            matched_keywords = sum(1 for kw in job_keywords if kw.lower() in resume_text.lower())
            ats_factors['keyword_usage'] = (matched_keywords / len(job_keywords)) * 100 if job_keywords else 0
        else:
            ats_factors['keyword_usage'] = 60  # Default medium score if no keywords given

    # 3. Avoiding complex formatting (assume plain text = good formatting)
    # This is basic since we can't parse formatting directly from raw text
        ats_factors['formatting'] = 100

    # 4. Presence of bullet points (commonly used in ATS readable resumes)
        bullet_points = len(re.findall(r"[-•]", resume_text))
        ats_factors['bullet_points'] = min(100, bullet_points * 5)

        # Length appropriateness
        word_count = len(resume_text.split())
        if 300 <= word_count <= 800:
            ats_factors['length'] = 100
        elif 200 <= word_count <= 1200:
            ats_factors['length'] = 80
        else:
            ats_factors['length'] = 60
    
        # Calculate overall ATS score
        overall_ats_score = sum(ats_factors.values()) / len(ats_factors)
        
        return {
            'overall_score': round(overall_ats_score, 2),
            'detailed_scores': ats_factors,
            'recommendations': self._generate_ats_recommendations(ats_factors)
        }
    
    def _extract_skills_from_data(self, data: Dict) -> Dict[str, List[str]]:
        """Extract skills from resume/JD data"""
        # This would integrate with the NLP utilities
        text = data.get('raw_text', data.get('text', ''))
        
        # Simplified skill extraction (would use AdvancedNLPProcessor in real implementation)
        skills = {
            'programming_languages': [],
            'web_technologies': [],
            'data_science': [],
            'cloud_devops': [],
            'databases': []
        }
        
        skill_patterns = {
            'programming_languages': [
                r'\bpython\b', r'\bjava\b', r'\bjavascript\b', r'\bc\+\+\b', r'\bc#\b'
            ],
            'web_technologies': [
                r'\breact\b', r'\bangular\b', r'\bvue\b', r'\bhtml\b', r'\bcss\b'
            ],
            'data_science': [
                r'\bpandas\b', r'\bnumpy\b', r'\bscikit-learn\b', r'\btensorflow\b'
            ],
            'cloud_devops': [
                r'\baws\b', r'\bazure\b', r'\bdocker\b', r'\bkubernetes\b'
            ],
            'databases': [
                r'\bmysql\b', r'\bpostgresql\b', r'\bmongodb\b', r'\bredis\b'
            ]
        }
        
        text_lower = text.lower()
        for category, patterns in skill_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    skill_name = pattern.replace(r'\b', '').replace('\\', '')
                    skills[category].append(skill_name)
        
        return {k: v for k, v in skills.items() if v}
    
    def _get_category_weight(self, category: str, job_requirements: Dict = None) -> float:
        """Get weight for skill category based on job requirements"""
        if not job_requirements:
            return 1.0
        
        industry = job_requirements.get('industry', 'general')
        
        if industry in self.industry_skill_weights:
            return self.industry_skill_weights[industry].get(category, 0.1)
        
        return 1.0
    
    def _extract_years_of_experience(self, text: str) -> int:
        """Extract years of experience from text"""
        patterns = [
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience',
            r'experience\s*:\s*(\d+)\s*\+?\s*years?',
            r'(\d+)\s*years?\s+in\s+\w+'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return int(matches[0])
        
        # Fallback: count job positions and estimate
        job_count = len(re.findall(r'\d{4}\s*[-–]\s*(?:\d{4}|present|current)', text, re.IGNORECASE))
        return max(0, job_count * 2)  # Estimate 2 years per position
    
    def _extract_required_experience(self, text: str) -> Optional[int]:
        """Extract required years of experience from JD"""
        patterns = [
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience\s+required',
            r'minimum\s+(?:of\s+)?(\d+)\s*\+?\s*years?',
            r'(\d+)\s*\+?\s*years?\s+(?:in|with|of)',
            r'requires?\s+(\d+)\s*\+?\s*years?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return int(matches[0])
        
        return None
    
    def _extract_education_level(self, text: str) -> str:
        """Extract highest education level from resume"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['phd', 'ph.d', 'doctorate', 'doctoral']):
            return 'phd'
        elif any(term in text_lower for term in ['master', 'm.s', 'm.a', 'mba', 'm.tech']):
            return 'master'
        elif any(term in text_lower for term in ['bachelor', 'b.s', 'b.a', 'b.tech', 'be']):
            return 'bachelor'
        elif any(term in text_lower for term in ['associate', 'diploma']):
            return 'associate'
        else:
            return 'high_school'
    
    def _extract_required_education(self, text: str) -> str:
        """Extract required education level from JD"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['phd', 'ph.d', 'doctorate required']):
            return 'phd'
        elif any(term in text_lower for term in ['master', 'masters degree', 'graduate degree']):
            return 'master'
        elif any(term in text_lower for term in ['bachelor', 'bachelors', 'college degree', 'university degree']):
            return 'bachelor'
        else:
            return 'bachelor'  # Default assumption
    
    def _extract_important_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract most important keywords from job description"""
        try:
            # Use TF-IDF to find important terms
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = tfidf_scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
            return keywords
            
        except Exception:
            # Fallback: simple word frequency
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            return sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:top_n]
    
    def _get_role_specific_weights(self, role_type: str, industry: str) -> Dict[str, float]:
        """Get role-specific component weights"""
        base_weights = {
            'technical_skills': 0.4,
            'experience': 0.3,
            'education': 0.15,
            'soft_skills': 0.1,
            'certifications': 0.05
        }
        
        # Adjust weights based on role type
        role_adjustments = {
            'software_engineer': {
                'technical_skills': 0.5,
                'experience': 0.3,
                'education': 0.1,
                'soft_skills': 0.05,
                'certifications': 0.05
            },
            'data_scientist': {
                'technical_skills': 0.45,
                'experience': 0.25,
                'education': 0.2,
                'soft_skills': 0.05,
                'certifications': 0.05
            },
            'product_manager': {
                'technical_skills': 0.2,
                'experience': 0.4,
                'education': 0.15,
                'soft_skills': 0.2,
                'certifications': 0.05
            },
            'marketing': {
                'technical_skills': 0.15,
                'experience': 0.35,
                'education': 0.15,
                'soft_skills': 0.25,
                'certifications': 0.1
            }
        }
        
        return role_adjustments.get(role_type, base_weights)
    
    def _calculate_component_score(
        self, 
        resume_data: Dict, 
        jd_data: Dict, 
        component: str, 
        job_requirements: Dict
    ) -> float:
        """Calculate score for a specific component"""
        
        if component == 'technical_skills':
            return self._calculate_skill_match_score(resume_data, jd_data, job_requirements)
        elif component == 'experience':
            return self._calculate_experience_match_score(resume_data, jd_data, job_requirements)
        elif component == 'education':
            return self._calculate_education_match_score(resume_data, jd_data, job_requirements)
        elif component == 'soft_skills':
            return self._calculate_soft_skills_score(resume_data, jd_data)
        elif component == 'certifications':
            return self._calculate_certifications_score(resume_data, jd_data)
        else:
            return 70.0  # Default score
    
    def _calculate_soft_skills_score(self, resume_data: Dict, jd_data: Dict) -> float:
        """Calculate soft skills matching score"""
        soft_skill_keywords = [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'analytical', 'creative', 'adaptable', 'collaborative',
            'self-motivated', 'detail-oriented', 'time management'
        ]
        
        resume_text = resume_data.get('raw_text', '').lower()
        jd_text = jd_data.get('raw_text', jd_data.get('text', '')).lower()
        
        # Find soft skills mentioned in JD
        jd_soft_skills = [skill for skill in soft_skill_keywords if skill in jd_text]
        
        if not jd_soft_skills:
            return 70.0  # Neutral score if no soft skills mentioned
        
        # Check how many are mentioned in resume
        resume_soft_skills = [skill for skill in jd_soft_skills if skill in resume_text]
        
        match_ratio = len(resume_soft_skills) / len(jd_soft_skills)
        return match_ratio * 100
    
    def _calculate_certifications_score(self, resume_data: Dict, jd_data: Dict) -> float:
        """Calculate certifications matching score"""
        cert_patterns = [
            r'certified', r'certification', r'license', r'credential',
            r'aws certified', r'microsoft certified', r'google certified',
            r'pmp', r'cissp', r'cisa', r'ceh'
        ]
        
        resume_text = resume_data.get('raw_text', '').lower()
        jd_text = jd_data.get('raw_text', jd_data.get('text', '')).lower()
        
        # Count certifications in each text
        resume_certs = sum(1 for pattern in cert_patterns if re.search(pattern, resume_text))
        jd_certs = sum(1 for pattern in cert_patterns if re.search(pattern, jd_text))
        
        if jd_certs == 0:
            return 70.0  # Neutral if no certifications mentioned
        
        if resume_certs >= jd_certs:
            return 100.0
        else:
            return (resume_certs / jd_certs) * 100
    
    def _calculate_confidence_interval(self, score: float, variance: float) -> Tuple[float, float]:
        """Calculate confidence interval for the score"""
        std_dev = np.sqrt(variance)
        margin_of_error = 1.96 * std_dev  # 95% confidence interval
        
        lower_bound = max(0, score - margin_of_error)
        upper_bound = min(100, score + margin_of_error)
        
        return (round(lower_bound, 2), round(upper_bound, 2))
    
    def generate_improvement_recommendations(
        self, 
        match_score: ScoreBreakdown, 
        resume_data: Dict, 
        jd_data: Dict
    ) -> List[Dict[str, str]]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        # Skill-based recommendations
        if match_score.skill_match_score < 70:
            missing_skills = self._identify_missing_skills(resume_data, jd_data)
            if missing_skills:
                recommendations.append({
                    'category': 'Skills',
                    'priority': 'High',
                    'recommendation': f'Add the following skills to your resume: {", ".join(missing_skills[:5])}',
                    'impact': 'High - Could improve match score by 15-25 points'
                })
        
        # Experience-based recommendations
        if match_score.experience_match_score < 60:
            recommendations.append({
                'category': 'Experience',
                'priority': 'High',
                'recommendation': 'Highlight relevant experience more prominently. Use specific examples and quantifiable achievements.',
                'impact': 'Medium - Could improve match score by 10-20 points'
            })
        
        # Education-based recommendations
        if match_score.education_match_score < 70:
            recommendations.append({
                'category': 'Education',
                'priority': 'Medium',
                'recommendation': 'Consider highlighting relevant coursework, certifications, or ongoing education.',
                'impact': 'Low - Could improve match score by 5-10 points'
            })
        
        # Content quality recommendations
        content_quality = match_score.detailed_breakdown.get('content_quality', 70)
        if content_quality < 60:
            recommendations.append({
                'category': 'Content Quality',
                'priority': 'Medium',
                'recommendation': 'Improve resume formatting, grammar, and structure. Ensure all essential sections are present.',
                'impact': 'Medium - Could improve overall presentation and ATS compatibility'
            })
        
        # Keyword optimization
        if match_score.keyword_density_score < 50:
            recommendations.append({
                'category': 'Keywords',
                'priority': 'High',
                'recommendation': 'Incorporate more keywords from the job description naturally throughout your resume.',
                'impact': 'High - Essential for ATS systems and initial screening'
            })
        
        return recommendations
    
    def _identify_missing_skills(self, resume_data: Dict, jd_data: Dict) -> List[str]:
        """Identify skills present in JD but missing from resume"""
        resume_skills = set()
        jd_skills = set()
        
        # Extract skills from both documents
        resume_skill_data = self._extract_skills_from_data(resume_data)
        jd_skill_data = self._extract_skills_from_data(jd_data)
        
        # Flatten skill lists
        for category_skills in resume_skill_data.values():
            resume_skills.update(category_skills)
        
        for category_skills in jd_skill_data.values():
            jd_skills.update(category_skills)
        
        # Find missing skills
        missing_skills = jd_skills - resume_skills
        return list(missing_skills)
    
    def calculate_ats_compatibility_score(self, resume_data: Dict) -> Dict[str, any]:
        """Calculate ATS (Applicant Tracking System) compatibility score"""
        
        ats_factors = {
            'file_format': 0,  # Assume PDF is good
            'section_headers': 0,
            'contact_info': 0,
            'keyword_usage': 0,
            'formatting': 0,
            'length': 0
        }
        
        resume_text = resume_data.get('raw_text', '')
        sections = resume_data.get('sections', {})
        contact_info = resume_data.get('contact_info', {})
        
        # File format (assuming PDF is processed correctly)
        ats_factors['file_format'] = 100
        
        # Section headers
        standard_sections = ['experience', 'education', 'skills', 'summary']
        present_sections = sum(1 for section in standard_sections if section in sections)
        ats_factors['section_headers'] = (present_sections / len(standard_sections)) * 100
        
        # Contact information
        essential_contact = ['email', 'phone']
        present_contact = sum(1 for field in essential_contact if field in contact_info)
        ats_factors['contact_info'] = (present_contact / len(essential_contact)) * 100
        
        # Keyword usage (density and variety)
        word_count = len(resume_text.split())
        unique_words = len(set(resume_text.lower().split()))
        keyword_variety = (unique_words / word_count) if word_count > 0 else 0
        ats_factors['keyword_usage'] = min(100, keyword_variety * 200)
        
        # Formatting (simple heuristics)
        formatting_score = 100
        if len(re.findall(r'[^\w\s-]', resume_text)) > word_count * 0.1:
            formatting_score -= 20  # Too many special characters
        
        ats_factors['formatting'] = formatting_score
        
        # Length appropriateness
        if 300 <= word_count <= 800:
            ats_factors['length'] = 100
        elif 200 <= word_count <= 1200:
            ats_factors['length'] = 80
        else:
            ats_factors['length'] = 60
        
        # Calculate overall ATS score
        overall_ats_score = sum(ats_factors.values()) / len(ats_factors)
        
        return {
            'overall_score': round(overall_ats_score, 2),
            'detailed_scores': ats_factors,
            'recommendations': self._generate_ats_recommendations(ats_factors)
        }
    
    def _generate_ats_recommendations(self, ats_factors: Dict[str, float]) -> List[str]:
        """Generate ATS-specific recommendations"""
        recommendations = []
        
        if ats_factors['section_headers'] < 80:
            recommendations.append("Use standard section headers like 'Experience', 'Education', 'Skills'")
        
        if ats_factors['contact_info'] < 100:
            recommendations.append("Ensure email and phone number are clearly visible")
        
        if ats_factors['keyword_usage'] < 60:
            recommendations.append("Include more industry-relevant keywords naturally")
        
        if ats_factors['formatting'] < 80:
            recommendations.append("Simplify formatting and reduce special characters")
        
        if ats_factors['length'] < 80:
            recommendations.append("Optimize resume length (aim for 1-2 pages / 300-800 words)")
        
        return recommendations