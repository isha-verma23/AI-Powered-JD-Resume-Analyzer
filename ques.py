"""
AI Interview Generator -  AI-Powered Resume Analyzer Pro
Copyright (c) 2025 Isha Verma. All rights reserved.
Version: 1.0

This software is the intellectual property of Isha Verma.
Unauthorized copying, distribution, or modification is prohibited.
"""


import random
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from textblob import TextBlob
import numpy as np

class QuestionType(Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SITUATIONAL = "situational"
    ROLE_SPECIFIC = "role_specific"
    COMPANY_CULTURE = "company_culture"

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class InterviewQuestion:
    question: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    skills_tested: List[str]
    ideal_answer_points: List[str]
    follow_up_questions: List[str]
    evaluation_criteria: Dict[str, float]

@dataclass
class AnswerAnalysis:
    content_score: float
    structure_score: float
    technical_accuracy: float
    communication_score: float
    confidence_level: float
    sentiment_analysis: Dict[str, float]
    improvement_suggestions: List[str]
    overall_rating: str

class AIInterviewGenerator:
    """
    AI-powered interview question generator with personalized content
    """
    
    def __init__(self):
        self.question_templates = self._load_question_templates()
        self.evaluation_rubrics = self._load_evaluation_rubrics()
        self.industry_contexts = self._load_industry_contexts()
        
        # Answer analysis patterns
        self.positive_indicators = [
            r'\b(?:achieved|accomplished|improved|increased|reduced|optimized|led|managed|developed|created|built|designed|implemented)\b',
            r'\b(?:successfully|effectively|efficiently|significantly)\b',
            r'\b(?:result|outcome|impact|benefit|solution)\b'
        ]
        
        self.structure_indicators = [
            r'\b(?:first|second|third|initially|then|next|finally|in conclusion)\b',
            r'\b(?:for example|for instance|specifically|such as)\b',
            r'\b(?:situation|task|action|result)\b'  # STAR method
        ]
        
        self.confidence_indicators = {
            'high': [r'\b(?:definitely|certainly|absolutely|confident|sure)\b'],
            'medium': [r'\b(?:believe|think|probably|likely)\b'],
            'low': [r'\b(?:maybe|perhaps|might|possibly|not sure|uncertain)\b']
        }
    
    def generate_personalized_questions(
        self,
        jd_data: Dict,
        resume_data: Dict,
        role_type: str = "general",
        interview_type: str = "general",
        difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM,
        question_count: int = 10
    ) -> List[InterviewQuestion]:
        """
        Generate personalized interview questions based on JD and resume analysis
        """
        
        # Analyze JD and resume for context
        context = self._analyze_context(jd_data, resume_data)
        
        # Generate question mix
        question_distribution = self._get_question_distribution(interview_type)
        
        questions = []
        
        for question_type, count in question_distribution.items():
            if len(questions) >= question_count:
                break
                
            type_questions = self._generate_questions_by_type(
                question_type, count, context, difficulty_level, jd_data, resume_data
            )
            questions.extend(type_questions)
        
        # Randomize and trim to requested count
        random.shuffle(questions)
        return questions[:question_count]
    
    def generate_interview_questions(self, jd_text: str, resume_text: str):
        jd_data = {"raw_text": jd_text}
        resume_data = {"raw_text": resume_text}
        return self.generate_personalized_questions(jd_data, resume_data)

    
    def _load_question_templates(self) -> Dict[str, List[Dict]]:
        """Load question templates for different categories"""
        return {
            "technical": [
                {
                    "template": "How would you approach {problem_type} in {technology}?",
                    "skills": ["problem_solving", "technical_knowledge"],
                    "follow_ups": ["What challenges might you face?", "How would you optimize this solution?"]
                },
                {
                    "template": "Explain the difference between {concept_a} and {concept_b}.",
                    "skills": ["conceptual_knowledge", "communication"],
                    "follow_ups": ["When would you use each approach?", "What are the trade-offs?"]
                },
                {
                    "template": "Walk me through how you would debug {issue_type} in a {system_type} system.",
                    "skills": ["debugging", "system_thinking"],
                    "follow_ups": ["What tools would you use?", "How would you prevent this in the future?"]
                }
            ],
            "behavioral": [
                {
                    "template": "Tell me about a time when you had to {situation} while working on {project_type}.",
                    "skills": ["experience", "problem_solving", "communication"],
                    "follow_ups": ["What did you learn from this?", "How would you handle it differently now?"]
                },
                {
                    "template": "Describe a situation where you had to {challenge} with limited {resource}.",
                    "skills": ["resourcefulness", "adaptability"],
                    "follow_ups": ["What was the outcome?", "How did this change your approach?"]
                },
                {
                    "template": "Give me an example of when you {leadership_action} in a team setting.",
                    "skills": ["leadership", "teamwork"],
                    "follow_ups": ["How did the team respond?", "What would you do differently?"]
                }
            ],
            "situational": [
                {
                    "template": "How would you handle a situation where {scenario} in our {work_environment}?",
                    "skills": ["judgment", "problem_solving"],
                    "follow_ups": ["What factors would influence your decision?", "What if the situation escalated?"]
                },
                {
                    "template": "If you were tasked with {task} but had {constraint}, what would be your approach?",
                    "skills": ["strategic_thinking", "resource_management"],
                    "follow_ups": ["How would you prioritize?", "What would you need to succeed?"]
                }
            ],
            "role_specific": [
                {
                    "template": "In this role, you'll be working with {technology_stack}. How would you {role_task}?",
                    "skills": ["role_knowledge", "technical_skills"],
                    "follow_ups": ["What's your experience with this stack?", "How would you get up to speed?"]
                }
            ]
        }
    
    def _load_evaluation_rubrics(self) -> Dict[str, Dict[str, float]]:
        """Load evaluation criteria for different question types"""
        return {
            "technical": {
                "technical_accuracy": 0.4,
                "problem_solving_approach": 0.3,
                "communication_clarity": 0.2,
                "depth_of_knowledge": 0.1
            },
            "behavioral": {
                "example_relevance": 0.3,
                "story_structure": 0.25,
                "outcome_focus": 0.25,
                "self_awareness": 0.2
            },
            "situational": {
                "judgment_quality": 0.35,
                "reasoning_process": 0.3,
                "practical_considerations": 0.2,
                "communication_style": 0.15
            }
        }
    
    def _load_industry_contexts(self) -> Dict[str, Dict]:
        """Load industry-specific contexts and vocabularies"""
        return {
            "technology": {
                "common_challenges": ["scalability", "performance", "security", "user experience"],
                "technologies": ["cloud platforms", "microservices", "APIs", "databases"],
                "methodologies": ["agile", "devops", "ci/cd", "testing"]
            },
            "finance": {
                "common_challenges": ["risk management", "compliance", "data accuracy", "reporting"],
                "technologies": ["trading systems", "risk models", "data analytics", "blockchain"],
                "methodologies": ["quantitative analysis", "stress testing", "audit procedures"]
            },
            "healthcare": {
                "common_challenges": ["patient safety", "data privacy", "regulatory compliance", "efficiency"],
                "technologies": ["EMR systems", "medical devices", "telemedicine", "AI diagnostics"],
                "methodologies": ["clinical protocols", "quality assurance", "patient care standards"]
            }
        }
    
    def _analyze_context(self, jd_data: Dict, resume_data: Dict) -> Dict:
        """Analyze JD and resume to extract context for question generation"""
        context = {
            "role_level": "mid",
            "industry": "technology",
            "key_skills": [],
            "experience_areas": [],
            "company_size": "medium",
            "work_style": "collaborative"
        }
        
        # Fixed: Proper handling of string data
        jd_text = ""
        if isinstance(jd_data, dict):
            jd_text = jd_data.get('raw_text', jd_data.get('text', ''))
        elif isinstance(jd_data, str):
            jd_text = jd_data
        jd_text = jd_text.lower()
        
        resume_text = ""
        if isinstance(resume_data, dict):
            resume_text = resume_data.get('raw_text', resume_data.get('text', ''))
        elif isinstance(resume_data, str):
            resume_text = resume_data
        resume_text = resume_text.lower()
        
        # Determine role level
        if any(term in jd_text for term in ['senior', 'lead', 'principal', 'architect']):
            context['role_level'] = 'senior'
        elif any(term in jd_text for term in ['junior', 'entry', 'associate', 'graduate']):
            context['role_level'] = 'junior'
        
        # Determine industry
        for industry, industry_data in self.industry_contexts.items():
            if any(term in jd_text for term in industry_data.get('technologies', [])):
                context['industry'] = industry
                break
        
        # Extract key skills
        skill_patterns = [
            r'\b(?:python|java|javascript|react|angular|aws|docker|kubernetes)\b',
            r'\b(?:machine learning|data science|analytics|sql|tableau)\b',
            r'\b(?:project management|leadership|communication|teamwork)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, jd_text)
            context['key_skills'].extend(matches)
        
        context['key_skills'] = list(set(context['key_skills']))[:10]  # Top 10 unique skills
        
        return context
    
    def _get_question_distribution(self, interview_type: str) -> Dict[QuestionType, int]:
        """Get question distribution based on interview type"""
        distributions = {
            "general": {
                QuestionType.TECHNICAL: 3,
                QuestionType.BEHAVIORAL: 3,
                QuestionType.SITUATIONAL: 2,
                QuestionType.ROLE_SPECIFIC: 2
            },
            "technical": {
                QuestionType.TECHNICAL: 6,
                QuestionType.BEHAVIORAL: 2,
                QuestionType.SITUATIONAL: 1,
                QuestionType.ROLE_SPECIFIC: 1
            },
            "behavioral": {
                QuestionType.BEHAVIORAL: 6,
                QuestionType.SITUATIONAL: 2,
                QuestionType.TECHNICAL: 1,
                QuestionType.ROLE_SPECIFIC: 1
            },
            "leadership": {
                QuestionType.BEHAVIORAL: 4,
                QuestionType.SITUATIONAL: 4,
                QuestionType.TECHNICAL: 1,
                QuestionType.ROLE_SPECIFIC: 1
            }
        }
        
        return distributions.get(interview_type, distributions["general"])
    
    def _generate_questions_by_type(
        self,
        question_type: QuestionType,
        count: int,
        context: Dict,
        difficulty: DifficultyLevel,
        jd_data: Dict,
        resume_data: Dict
    ) -> List[InterviewQuestion]:
        """Generate questions for a specific type"""
        
        questions = []
        templates = self.question_templates.get(question_type.value, [])
        
        for _ in range(count):
            if not templates:
                break
                
            template_data = random.choice(templates)
            
            # Personalize the template
            personalized_question = self._personalize_template(
                template_data, context, difficulty, jd_data, resume_data
            )
            
            if personalized_question:
                questions.append(personalized_question)
        
        return questions
    
    def _personalize_template(
        self,
        template_data: Dict,
        context: Dict,
        difficulty: DifficultyLevel,
        jd_data: Dict,
        resume_data: Dict
    ) -> Optional[InterviewQuestion]:
        """Personalize a question template with context-specific information"""
        
        template = template_data["template"]
        
        # Define replacement values based on context
        replacements = {
            "technology": random.choice(context.get("key_skills", ["Python", "JavaScript"])),
            "problem_type": self._get_problem_type(context, difficulty),
            "concept_a": self._get_concept_pair(context)[0],
            "concept_b": self._get_concept_pair(context)[1],
            "issue_type": self._get_issue_type(context, difficulty),
            "system_type": self._get_system_type(context),
            "situation": self._get_behavioral_situation(context, difficulty),
            "project_type": self._get_project_type(context),
            "challenge": self._get_challenge_type(context, difficulty),
            "resource": self._get_resource_constraint(context),
            "leadership_action": self._get_leadership_action(context, difficulty),
            "scenario": self._get_situational_scenario(context, difficulty),
            "work_environment": self._get_work_environment(context),
            "task": self._get_role_task(context, difficulty),
            "constraint": self._get_constraint_type(context),
            "technology_stack": self._get_technology_stack(jd_data),
            "role_task": self._get_specific_role_task(jd_data, context)
        }
        
        # Replace placeholders in template
        question_text = template
        for placeholder, value in replacements.items():
            question_text = question_text.replace(f"{{{placeholder}}}", value)
        
        # Generate follow-up questions
        follow_ups = self._generate_follow_up_questions(template_data, context, difficulty)
        
        # Create evaluation criteria
        evaluation_criteria = self._create_evaluation_criteria(template_data, difficulty)
        
        return InterviewQuestion(
            question=question_text,
            question_type=self._get_question_type_from_template(template_data),
            difficulty=difficulty,
            skills_tested=template_data.get("skills", []),
            ideal_answer_points=self._generate_ideal_answer_points(question_text, context),
            follow_up_questions=follow_ups,
            evaluation_criteria=evaluation_criteria
        )
    
    def _get_problem_type(self, context: Dict, difficulty: DifficultyLevel) -> str:
        """Get appropriate problem type based on context and difficulty"""
        problems = {
            "easy": ["data validation", "user input handling", "basic optimization"],
            "medium": ["performance bottlenecks", "scalability issues", "integration challenges"],
            "hard": ["distributed system failures", "complex algorithm optimization", "security vulnerabilities"]
        }
        
        return random.choice(problems.get(difficulty.value, problems["medium"]))
    
    def _get_concept_pair(self, context: Dict) -> Tuple[str, str]:
        """Get related concept pairs for comparison questions"""
        pairs = [
            ("REST APIs", "GraphQL"),
            ("SQL", "NoSQL"),
            ("synchronous", "asynchronous"),
            ("microservices", "monolithic architecture"),
            ("unit testing", "integration testing")
        ]
        
        return random.choice(pairs)
    
    def _get_issue_type(self, context: Dict, difficulty: DifficultyLevel) -> str:
        """Get issue type for debugging questions"""
        issues = {
            "easy": ["null pointer exceptions", "syntax errors", "configuration issues"],
            "medium": ["memory leaks", "race conditions", "API timeouts"],
            "hard": ["distributed transaction failures", "complex concurrency bugs", "performance degradation"]
        }
        
        return random.choice(issues.get(difficulty.value, issues["medium"]))
    
    def _get_system_type(self, context: Dict) -> str:
        """Get system type based on context"""
        systems = ["web application", "mobile app", "microservices", "database", "distributed system"]
        return random.choice(systems)
    
    def _get_behavioral_situation(self, context: Dict, difficulty: DifficultyLevel) -> str:
        """Get behavioral situation based on difficulty"""
        situations = {
            "easy": ["work with a difficult teammate", "meet a tight deadline", "learn a new technology"],
            "medium": ["handle conflicting priorities", "lead a failing project", "resolve team conflicts"],
            "hard": ["manage a crisis situation", "turn around a failing product", "handle major system outage"]
        }
        
        return random.choice(situations.get(difficulty.value, situations["medium"]))
    
    def _get_project_type(self, context: Dict) -> str:
        """Get project type based on context"""
        projects = ["web application", "mobile app", "data pipeline", "API integration", "machine learning model"]
        return random.choice(projects)
    
    def _get_challenge_type(self, context: Dict, difficulty: DifficultyLevel) -> str:
        """Get challenge type for situational questions"""
        challenges = {
            "easy": ["deliver a feature quickly", "fix a critical bug", "onboard a new team member"],
            "medium": ["migrate legacy systems", "optimize system performance", "implement new architecture"],
            "hard": ["redesign entire platform", "handle major security breach", "scale to 10x traffic"]
        }
        
        return random.choice(challenges.get(difficulty.value, challenges["medium"]))
    
    def _get_resource_constraint(self, context: Dict) -> str:
        """Get resource constraint for situational questions"""
        constraints = ["time", "budget", "team members", "technical resources", "information"]
        return random.choice(constraints)
    
    def _get_leadership_action(self, context: Dict, difficulty: DifficultyLevel) -> str:
        """Get leadership action based on difficulty"""
        actions = {
            "easy": ["mentored a junior developer", "led a small project", "facilitated team meetings"],
            "medium": ["managed project deadlines", "resolved team conflicts", "implemented process improvements"],
            "hard": ["led organizational change", "managed crisis response", "transformed team culture"]
        }
        
        return random.choice(actions.get(difficulty.value, actions["medium"]))
    
    def _get_situational_scenario(self, context: Dict, difficulty: DifficultyLevel) -> str:
        """Get situational scenario based on difficulty"""
        scenarios = {
            "easy": ["a team member is consistently late with deliverables", "you discover a bug in production"],
            "medium": ["your team is behind schedule on a critical project", "there's disagreement about technical approach"],
            "hard": ["the entire system goes down during peak hours", "a major client threatens to leave"]
        }
        
        return random.choice(scenarios.get(difficulty.value, scenarios["medium"]))
    
    def _get_work_environment(self, context: Dict) -> str:
        """Get work environment based on context"""
        environments = ["fast-paced startup", "established enterprise", "agile team", "remote-first company", "cross-functional team"]
        return random.choice(environments)
    
    def _get_role_task(self, context: Dict, difficulty: DifficultyLevel) -> str:
        """Get role-specific task based on difficulty"""
        tasks = {
            "easy": ["implement a new feature", "fix existing bugs", "write unit tests"],
            "medium": ["design system architecture", "optimize application performance", "mentor junior developers"],
            "hard": ["lead technical strategy", "architect scalable solutions", "drive innovation initiatives"]
        }
        
        return random.choice(tasks.get(difficulty.value, tasks["medium"]))
    
    def _get_constraint_type(self, context: Dict) -> str:
        """Get constraint type for questions"""
        constraints = ["limited budget", "tight timeline", "small team", "legacy technology", "regulatory requirements"]
        return random.choice(constraints)
    
    def _get_technology_stack(self, jd_data: Dict) -> str:
        """Extract technology stack from job description"""
        # Fixed: Proper handling of string data
        jd_text = ""
        if isinstance(jd_data, dict):
            jd_text = jd_data.get('raw_text', jd_data.get('text', ''))
        elif isinstance(jd_data, str):
            jd_text = jd_data
        jd_text = jd_text.lower()
        
        # Look for common tech stacks
        stacks = [
            "React/Node.js", "Python/Django", "Java/Spring", "Angular/.NET", 
            "Vue.js/Express", "PHP/Laravel", "Ruby on Rails", "Go/Gin"
        ]
        
        # Try to match from JD text or return random stack
        for stack in stacks:
            if any(tech.lower() in jd_text for tech in stack.split('/')):
                return stack
        
        return random.choice(stacks)
    
    def _get_specific_role_task(self, jd_data: Dict, context: Dict) -> str:
        """Get specific role task from job description"""
        # Fixed: Proper handling of string data
        jd_text = ""
        if isinstance(jd_data, dict):
            jd_text = jd_data.get('raw_text', jd_data.get('text', ''))
        elif isinstance(jd_data, str):
            jd_text = jd_data
        jd_text = jd_text.lower()
        
        # Common role tasks
        tasks = [
            "build scalable web applications", "develop REST APIs", "implement data pipelines",
            "optimize database queries", "design system architecture", "automate deployment processes"
        ]
        
        return random.choice(tasks)
    
    def _generate_follow_up_questions(self, template_data: Dict, context: Dict, difficulty: DifficultyLevel) -> List[str]:
        """Generate contextual follow-up questions"""
        base_follow_ups = template_data.get("follow_ups", [])
        
        # Add difficulty-specific follow-ups
        if difficulty == DifficultyLevel.HARD:
            base_follow_ups.extend([
                "How would you handle this at scale?",
                "What are the long-term implications?",
                "How would you measure success?"
            ])
        
        return base_follow_ups[:3]  # Limit to 3 follow-ups
    
    def _create_evaluation_criteria(self, template_data: Dict, difficulty: DifficultyLevel) -> Dict[str, float]:
        """Create evaluation criteria based on template and difficulty"""
        base_criteria = {
            "clarity": 0.25,
            "depth": 0.25,
            "relevance": 0.25,
            "structure": 0.25
        }
        
        # Adjust weights based on difficulty
        if difficulty == DifficultyLevel.HARD:
            base_criteria["depth"] = 0.35
            base_criteria["clarity"] = 0.20
        elif difficulty == DifficultyLevel.EASY:
            base_criteria["clarity"] = 0.35
            base_criteria["depth"] = 0.15
        
        return base_criteria
    
    def _generate_ideal_answer_points(self, question_text: str, context: Dict) -> List[str]:
        """Generate ideal answer points for a question"""
        # Basic answer points that apply to most questions
        points = [
            "Clear understanding of the problem/situation",
            "Structured approach to problem-solving",
            "Relevant examples or experience",
            "Consideration of trade-offs and alternatives"
        ]
        
        # Add context-specific points based on question content
        if "technical" in question_text.lower() or any(tech in question_text.lower() for tech in ["code", "system", "design"]):
            points.extend([
                "Technical accuracy and depth",
                "Best practices and industry standards",
                "Scalability and performance considerations"
            ])
        
        if any(word in question_text.lower() for word in ["team", "conflict", "leadership", "manage"]):
            points.extend([
                "Interpersonal skills demonstration",
                "Leadership and communication abilities",
                "Conflict resolution strategies"
            ])
        
        return points[:5]  # Limit to 5 key points
    
    def _get_question_type_from_template(self, template_data: Dict) -> QuestionType:
        """Determine question type from template data"""
        skills = template_data.get("skills", [])
        
        if any(skill in ["technical_knowledge", "debugging", "system_thinking"] for skill in skills):
            return QuestionType.TECHNICAL
        elif any(skill in ["leadership", "teamwork", "experience"] for skill in skills):
            return QuestionType.BEHAVIORAL
        elif any(skill in ["judgment", "strategic_thinking"] for skill in skills):
            return QuestionType.SITUATIONAL
        else:
            return QuestionType.ROLE_SPECIFIC

    def analyze_answer(self, question: InterviewQuestion, answer_text: str) -> AnswerAnalysis:
        """
        Analyze candidate's answer and provide detailed feedback
        """
        
        # Basic text analysis
        blob = TextBlob(answer_text)
        word_count = len(answer_text.split())
        
        # Content analysis
        content_score = self._analyze_content(question, answer_text, blob)
        
        # Structure analysis
        structure_score = self._analyze_structure(answer_text)
        
        # Technical accuracy (for technical questions)
        technical_accuracy = self._analyze_technical_accuracy(question, answer_text)
        
        # Communication analysis
        communication_score = self._analyze_communication(answer_text, blob)
        
        # Confidence analysis
        confidence_level = self._analyze_confidence(answer_text)
        
        # Sentiment analysis
        sentiment_analysis = {
            'polarity': float(blob.sentiment.polarity),
            'subjectivity': float(blob.sentiment.subjectivity)
        }
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            question, answer_text, content_score, structure_score, communication_score
        )
        
        # Calculate overall rating
        overall_rating = self._calculate_overall_rating(
            content_score, structure_score, technical_accuracy, communication_score, confidence_level
        )
        
        return AnswerAnalysis(
            content_score=content_score,
            structure_score=structure_score,
            technical_accuracy=technical_accuracy,
            communication_score=communication_score,
            confidence_level=confidence_level,
            sentiment_analysis=sentiment_analysis,
            improvement_suggestions=improvement_suggestions,
            overall_rating=overall_rating
        )
    
    def _analyze_content(self, question: InterviewQuestion, answer_text: str, blob: TextBlob) -> float:
        """Analyze content relevance and depth"""
        score = 0.0
        answer_lower = answer_text.lower()
        
        # Check for positive indicators
        for pattern in self.positive_indicators:
            matches = len(re.findall(pattern, answer_lower, re.IGNORECASE))
            score += min(matches * 0.1, 0.3)  # Cap contribution
        
        # Check for ideal answer points coverage
        ideal_points_covered = 0
        for point in question.ideal_answer_points:
            point_keywords = point.lower().split()
            if any(keyword in answer_lower for keyword in point_keywords):
                ideal_points_covered += 1
        
        if question.ideal_answer_points:
            score += (ideal_points_covered / len(question.ideal_answer_points)) * 0.4
        
        # Length consideration (optimal range: 100-300 words)
        word_count = len(answer_text.split())
        if 100 <= word_count <= 300:
            score += 0.2
        elif word_count > 50:
            score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_structure(self, answer_text: str) -> float:
        """Analyze answer structure and organization"""
        score = 0.0
        answer_lower = answer_text.lower()
        
        # Check for structure indicators
        for pattern in self.structure_indicators:
            if re.search(pattern, answer_lower, re.IGNORECASE):
                score += 0.2
        
        # Check for STAR method (Situation, Task, Action, Result)
        star_indicators = ['situation', 'task', 'action', 'result']
        star_count = sum(1 for indicator in star_indicators if indicator in answer_lower)
        if star_count >= 3:
            score += 0.3
        elif star_count >= 2:
            score += 0.2
        
        # Check for logical flow (paragraphs or clear sections)
        paragraphs = answer_text.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_technical_accuracy(self, question: InterviewQuestion, answer_text: str) -> float:
        """Analyze technical accuracy for technical questions"""
        if question.question_type != QuestionType.TECHNICAL:
            return 1.0  # Not applicable for non-technical questions
        
        score = 0.5  # Base score
        answer_lower = answer_text.lower()
        
        # Look for technical terms and concepts
        technical_terms = [
            'algorithm', 'data structure', 'complexity', 'optimization', 'scalability',
            'database', 'api', 'framework', 'library', 'architecture', 'design pattern'
        ]
        
        term_count = sum(1 for term in technical_terms if term in answer_lower)
        score += min(term_count * 0.05, 0.3)
        
        # Check for specific technical skills mentioned in question
        for skill in question.skills_tested:
            if skill.replace('_', ' ') in answer_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_communication(self, answer_text: str, blob: TextBlob) -> float:
        """Analyze communication clarity and effectiveness"""
        score = 0.0
        
        # Grammar and readability (simplified)
        sentences = blob.sentences
        if sentences:
            avg_sentence_length = sum(len(str(s).split()) for s in sentences) / len(sentences)
            if 10 <= avg_sentence_length <= 25:  # Optimal sentence length
                score += 0.3
            else:
                score += 0.1
        
        # Vocabulary diversity
        words = answer_text.lower().split()
        unique_words = set(words)
        if words:
            diversity_ratio = len(unique_words) / len(words)
            score += min(diversity_ratio * 0.5, 0.3)
        
        # Professional tone indicators
        professional_indicators = ['experience', 'project', 'team', 'solution', 'approach', 'strategy']
        professional_count = sum(1 for indicator in professional_indicators if indicator in answer_text.lower())
        score += min(professional_count * 0.05, 0.25)
        
        # Clear examples and specific details
        example_indicators = ['for example', 'for instance', 'specifically', 'in particular']
        if any(indicator in answer_text.lower() for indicator in example_indicators):
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_confidence(self, answer_text: str) -> float:
        """Analyze confidence level in the answer"""
        answer_lower = answer_text.lower()
        confidence_score = 0.5  # Neutral baseline
        
        # High confidence indicators
        for pattern in self.confidence_indicators['high']:
            matches = len(re.findall(pattern, answer_lower, re.IGNORECASE))
            confidence_score += matches * 0.1
        
        # Medium confidence indicators
        for pattern in self.confidence_indicators['medium']:
            matches = len(re.findall(pattern, answer_lower, re.IGNORECASE))
            confidence_score += matches * 0.05
        
        # Low confidence indicators (reduce score)
        for pattern in self.confidence_indicators['low']:
            matches = len(re.findall(pattern, answer_lower, re.IGNORECASE))
            confidence_score -= matches * 0.1
        
        return max(0.0, min(confidence_score, 1.0))
    
    def _generate_improvement_suggestions(
        self,
        question: InterviewQuestion,
        answer_text: str,
        content_score: float,
        structure_score: float,
        communication_score: float
    ) -> List[str]:
        """Generate personalized improvement suggestions"""
        suggestions = []
        
        # Content improvements
        if content_score < 0.6:
            suggestions.append("Provide more specific examples and concrete details to support your points")
            suggestions.append("Address more aspects of the question to demonstrate comprehensive understanding")
        
        # Structure improvements
        if structure_score < 0.6:
            suggestions.append("Use a clearer structure like the STAR method (Situation, Task, Action, Result)")
            suggestions.append("Organize your response with clear transitions between ideas")
        
        # Communication improvements
        if communication_score < 0.6:
            suggestions.append("Use more professional vocabulary and industry-specific terms")
            suggestions.append("Vary your sentence structure for better readability")
        
        # Question-specific suggestions
        if question.question_type == QuestionType.TECHNICAL:
            if 'algorithm' not in answer_text.lower() and 'approach' not in answer_text.lower():
                suggestions.append("Explain your problem-solving approach and methodology")
        
        if question.question_type == QuestionType.BEHAVIORAL:
            if not any(indicator in answer_text.lower() for indicator in ['result', 'outcome', 'impact']):
                suggestions.append("Clearly state the results and impact of your actions")
        
        # General suggestions if few specific ones
        if len(suggestions) < 2:
            suggestions.extend([
                "Practice articulating your thoughts more clearly and concisely",
                "Prepare specific examples that demonstrate your skills and experience"
            ])
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _calculate_overall_rating(
        self,
        content_score: float,
        structure_score: float,
        technical_accuracy: float,
        communication_score: float,
        confidence_level: float
    ) -> str:
        """Calculate overall performance rating"""
        
        # Weighted average
        overall_score = (
            content_score * 0.3 +
            structure_score * 0.25 +
            technical_accuracy * 0.2 +
            communication_score * 0.15 +
            confidence_level * 0.1
        )
        
        if overall_score >= 0.85:
            return "Excellent"
        elif overall_score >= 0.7:
            return "Good"
        elif overall_score >= 0.55:
            return "Average"
        elif overall_score >= 0.4:
            return "Below Average"
        else:
            return "Poor"

    def get_question_bank(self, filters: Dict = None) -> List[InterviewQuestion]:
        """
        Get a filtered bank of pre-generated questions
        """
        if filters is None:
            filters = {}
        
        # This would typically load from a database or file
        # For now, generate a sample set
        sample_context = {
            "role_level": "mid",
            "industry": "technology",
            "key_skills": ["Python", "JavaScript", "React", "SQL"],
            "experience_areas": ["web development", "data analysis"],
            "company_size": "medium",
            "work_style": "collaborative"
        }
        
        sample_jd = {"raw_text": "Senior Software Engineer position requiring Python, React, and cloud experience"}
        sample_resume = {"raw_text": "Software engineer with 5 years experience in web development"}
        
        all_questions = []
        
        # Generate questions for different types and difficulties
        for question_type in QuestionType:
            for difficulty in DifficultyLevel:
                questions = self._generate_questions_by_type(
                    question_type, 3, sample_context, difficulty, sample_jd, sample_resume
                )
                all_questions.extend(questions)
        
        # Apply filters
        filtered_questions = all_questions
        
        if filters.get('question_type'):
            filtered_questions = [q for q in filtered_questions if q.question_type == filters['question_type']]
        
        if filters.get('difficulty'):
            filtered_questions = [q for q in filtered_questions if q.difficulty == filters['difficulty']]
        
        if filters.get('skills'):
            skill_filter = filters['skills']
            filtered_questions = [q for q in filtered_questions 
                                if any(skill in q.skills_tested for skill in skill_filter)]
        
        return filtered_questions
    
    def export_questions(self, questions: List[InterviewQuestion], format: str = "json") -> str:
        """
        Export questions in specified format
        """
        if format == "json":
            return self._export_to_json(questions)
        elif format == "csv":
            return self._export_to_csv(questions)
        elif format == "txt":
            return self._export_to_txt(questions)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_to_json(self, questions: List[InterviewQuestion]) -> str:
        """Export questions to JSON format"""
        questions_data = []
        for q in questions:
            question_dict = {
                "question": q.question,
                "type": q.question_type.value,
                "difficulty": q.difficulty.value,
                "skills_tested": q.skills_tested,
                "ideal_answer_points": q.ideal_answer_points,
                "follow_up_questions": q.follow_up_questions,
                "evaluation_criteria": q.evaluation_criteria
            }
            questions_data.append(question_dict)
        
        return json.dumps(questions_data, indent=2)
    
    def _export_to_csv(self, questions: List[InterviewQuestion]) -> str:
        """Export questions to CSV format"""
        csv_lines = ["Question,Type,Difficulty,Skills,Follow-ups"]
        
        for q in questions:
            skills_str = "; ".join(q.skills_tested)
            follow_ups_str = "; ".join(q.follow_up_questions)
            
            # Escape commas and quotes in the question text
            question_clean = q.question.replace('"', '""')
            
            csv_line = f'"{question_clean}",{q.question_type.value},{q.difficulty.value},"{skills_str}","{follow_ups_str}"'
            csv_lines.append(csv_line)
        
        return "\n".join(csv_lines)
    
    def _export_to_txt(self, questions: List[InterviewQuestion]) -> str:
        """Export questions to plain text format"""
        txt_lines = []
        
        for i, q in enumerate(questions, 1):
            txt_lines.append(f"Question {i}:")
            txt_lines.append(f"Type: {q.question_type.value.title()}")
            txt_lines.append(f"Difficulty: {q.difficulty.value.title()}")
            txt_lines.append(f"Question: {q.question}")
            txt_lines.append(f"Skills Tested: {', '.join(q.skills_tested)}")
            txt_lines.append("Follow-up Questions:")
            for follow_up in q.follow_up_questions:
                txt_lines.append(f"  - {follow_up}")
            txt_lines.append("Ideal Answer Points:")
            for point in q.ideal_answer_points:
                txt_lines.append(f"  - {point}")
            txt_lines.append("-" * 80)
            txt_lines.append("")
        
        return "\n".join(txt_lines)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the generator
    generator = AIInterviewGenerator()
    
    # Sample job description and resume data
    sample_jd = {
        "raw_text": """
        Senior Software Engineer - Full Stack Development
        
        We are looking for an experienced Senior Software Engineer to join our growing team.
        You will be responsible for developing scalable web applications using React, Node.js,
        and Python. Experience with AWS, Docker, and microservices architecture is required.
        
        Requirements:
        - 5+ years of software development experience
        - Proficiency in React, Node.js, Python
        - Experience with cloud platforms (AWS preferred)
        - Strong problem-solving and communication skills
        - Experience with agile development methodologies
        """
    }
    
    sample_resume = {
        "raw_text": """
        John Doe - Software Engineer
        
        Experience:
        - Software Engineer at TechCorp (3 years)
          - Developed web applications using React and Node.js
          - Implemented REST APIs and microservices
          - Worked with AWS and Docker for deployment
        
        - Junior Developer at StartupXYZ (2 years)
          - Built frontend components using JavaScript and React
          - Collaborated with cross-functional teams
        
        Skills: JavaScript, React, Node.js, Python, AWS, Docker, SQL
        """
    }
    
    # Generate personalized questions
    try:
        questions = generator.generate_personalized_questions(
            jd_data=sample_jd,
            resume_data=sample_resume,
            interview_type="general",
            difficulty_level=DifficultyLevel.MEDIUM,
            question_count=5
        )
        
        print("Generated Interview Questions:")
        print("=" * 50)
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i} ({question.question_type.value.title()} - {question.difficulty.value.title()}):")
            print(f"Q: {question.question}")
            print(f"Skills tested: {', '.join(question.skills_tested)}")
            print(f"Follow-ups: {', '.join(question.follow_up_questions)}")
        
        # Example answer analysis
        sample_answer = """
        In my previous role at TechCorp, I encountered a similar scalability issue when our web application
        started experiencing slow response times during peak hours. First, I analyzed the system performance
        using monitoring tools and identified that database queries were the bottleneck. I implemented
        database indexing and query optimization, which improved response times by 40%. Additionally,
        I introduced Redis caching for frequently accessed data. The result was a much more responsive
        application that could handle 3x more concurrent users.
        """
        
        if questions:
            analysis = generator.analyze_answer(questions[0], sample_answer)
            
            print(f"\nSample Answer Analysis:")
            print(f"Content Score: {analysis.content_score:.2f}")
            print(f"Structure Score: {analysis.structure_score:.2f}")
            print(f"Communication Score: {analysis.communication_score:.2f}")
            print(f"Overall Rating: {analysis.overall_rating}")
            print("Improvement Suggestions:")
            for suggestion in analysis.improvement_suggestions:
                print(f"  - {suggestion}")
        
        # Export questions
        json_export = generator.export_questions(questions, "json")
        print(f"\nExported {len(questions)} questions to JSON format")
        
        # Get question bank with filters
        filtered_questions = generator.get_question_bank({
            'question_type': QuestionType.TECHNICAL,
            'difficulty': DifficultyLevel.HARD
        })
        print(f"\nFound {len(filtered_questions)} technical hard questions in question bank")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        