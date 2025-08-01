"""
AI Interview Generator -  AI-Powered Resume Analyzer Pro
Copyright (c) 2025 Isha Verma. All rights reserved.
Version: 1.0

This software is the intellectual property of Isha Verma.
Unauthorized copying, distribution, or modification is prohibited.
"""

import random
import fitz  # PyMuPDF
import re
import io
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import nltk
print("NLTK data search paths:", nltk.data.path)

# Add path explicitly if needed (replace this with your actual nltk_data folder)
nltk.data.path.append('/Users/youruser/nltk_data')

# Download both punkt and punkt_tab just to cover both resources
nltk.download('punkt')
nltk.download('punkt_tab')

@dataclass
class ExtractedSection:
    title: str
    content: str
    page_number: int
    confidence: float

class AdvancedPDFParser:
    """
    Advanced PDF parser with section detection and robust text extraction
    (+ OCR fallback for image-based PDFs)
    """

    def __init__(self):
        self.section_patterns = {
            'contact': [
                r'contact\s+information',
                r'personal\s+details',
                r'contact\s+details'
            ],
            'summary': [
                r'professional\s+summary',
                r'career\s+summary',
                r'profile',
                r'objective',
                r'summary',
                r'about\s+me'
            ],
            'experience': [
                r'work\s+experience',
                r'professional\s+experience',
                r'employment\s+history',
                r'career\s+history',
                r'experience'
            ],
            'education': [
                r'education',
                r'academic\s+background',
                r'educational\s+background',
                r'qualifications'
            ],
            'skills': [
                r'technical\s+skills',
                r'skills',
                r'core\s+competencies',
                r'expertise',
                r'technologies'
            ],
            'projects': [
                r'projects',
                r'key\s+projects',
                r'notable\s+projects',
                r'personal\s+projects'
            ],
            'certifications': [
                r'certifications',
                r'certificates',
                r'professional\s+certifications',
                r'licenses'
            ],
            'achievements': [
                r'achievements',
                r'accomplishments',
                r'awards',
                r'honors'
            ]
        }
        self.layout_indicators = {
            'single_column': ['traditional', 'classic', 'simple'],
            'two_column': ['modern', 'creative', 'sidebar'],
            'multi_section': ['academic', 'detailed', 'comprehensive']
        }

    def extract_text_from_pdf(self, pdf_file: Union[str, io.IOBase]) -> Dict[str, any]:
        """
        Extract text from PDF with enhanced structure detection and OCR fallback
        """
        try:
            pdf_data = self._load_pdf_binary(pdf_file)
            if pdf_data is None:
                raise ValueError("Could not read PDF as binary data.")
            
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

            extraction_result = {
                'raw_text': '',
                'sections': {},
                'metadata': {
                    'page_count': pdf_document.page_count,
                    'layout_type': 'unknown',
                    'extraction_confidence': 0.0
                },
                'contact_info': {},
                'structure_analysis': {}
            }

            all_text = ""
            page_texts = []

            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text_methods = [
                    self._extract_with_layout_preservation(page),
                    self._extract_with_blocks(page),
                    page.get_text()  # Fallback
                ]
                best_text = ""
                best_score = 0
                for text in text_methods:
                    if text and len(text.strip()) > len(best_text.strip()):
                        score = self._evaluate_text_quality(text)
                        if score > best_score:
                            best_text = text
                            best_score = score
                page_texts.append(best_text)
                all_text += best_text + "\n"

            pdf_document.close()

            # Preprocess to fix section headers split over newlines (JOIN W\nO\nR\nK to WORK)
            all_text = re.sub(r'((?:[A-Za-z]\n){2,}[A-Za-z])',
                              lambda m: m.group().replace("\n", ""), all_text)
            # Also fix split words (e.g., "WORK\nEXPERIENCE" => "WORK EXPERIENCE")
            all_text = re.sub(r'([A-Za-z])\n([A-Za-z])', r'\1 \2', all_text)

            extraction_result['raw_text'] = self._clean_extracted_text(all_text)
            extraction_result['sections'] = self._detect_sections(extraction_result['raw_text'])
            extraction_result['contact_info'] = self._extract_contact_info(extraction_result['raw_text'])
            extraction_result['structure_analysis'] = self._analyze_document_structure(
                extraction_result['raw_text'], page_texts)
            extraction_result['metadata']['extraction_confidence'] = self._calculate_extraction_confidence(
                extraction_result)

            # If text extraction completely failed, try OCR fallback
            if not extraction_result['raw_text'].strip():
                try:
                    ocr_text = self._extract_text_with_ocr(pdf_data)
                    if ocr_text.strip():
                        ocr_text = re.sub(r'((?:[A-Za-z]\n){2,}[A-Za-z])',
                                          lambda m: m.group().replace("\n", ""), ocr_text)
                        ocr_text = re.sub(r'([A-Za-z])\n([A-Za-z])', r'\1 \2', ocr_text)
                        extraction_result['raw_text'] = self._clean_extracted_text(ocr_text)
                        extraction_result['sections'] = self._detect_sections(extraction_result['raw_text'])
                        extraction_result['contact_info'] = self._extract_contact_info(extraction_result['raw_text'])
                        extraction_result['structure_analysis'] = self._analyze_document_structure(
                            extraction_result['raw_text'], [ocr_text])
                        extraction_result['metadata']['extraction_confidence'] = self._calculate_extraction_confidence(
                            extraction_result)
                except Exception as ocr_e:
                    logging.error(f"OCR fallback failed: {ocr_e}")

            return extraction_result

        except Exception as e:
            logging.error(f"Error extracting text from PDF: {str(e)}")
            return {
                'raw_text': '',
                'sections': {},
                'metadata': {'page_count': 0, 'layout_type': 'unknown', 'extraction_confidence': 0.0},
                'contact_info': {},
                'structure_analysis': {},
                'error': str(e)
            }

    def _load_pdf_binary(self, pdf_file: Union[str, io.IOBase]) -> Optional[bytes]:
        """Load PDF as binary from file path or stream"""
        if isinstance(pdf_file, str):
            with open(pdf_file, 'rb') as f:
                return f.read()
        elif hasattr(pdf_file, 'read'):
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
            return pdf_file.read()
        return None

    def _extract_with_layout_preservation(self, page) -> str:
        """Extract text while preserving layout structure"""
        try:
            blocks = page.get_text("dict")["blocks"]
            text_blocks = []
            for block in blocks:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        block_text += line_text + "\n"
                    if block_text.strip():
                        text_blocks.append({
                            'text': block_text.strip(),
                            'bbox': block["bbox"]
                        })
            text_blocks.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
            combined_text = "\n".join([block['text'] for block in text_blocks])
            return combined_text
        except Exception:
            return ""

    def _extract_with_blocks(self, page) -> str:
        """Extract text using block-based method and join blocks"""
        try:
            blocks = page.get_text("blocks")
            block_text = [b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)]
            return "\n".join(block_text)
        except Exception:
            return ""

    def _extract_text_with_ocr(self, pdf_bytes: bytes) -> str:
        """Convert PDF to images & extract text using OCR"""
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
        except ImportError as e:
            raise ImportError("OCR fallback requires pdf2image and pytesseract.") from e
        images = convert_from_bytes(pdf_bytes)
        return "\n".join(pytesseract.image_to_string(img) for img in images)

    def _evaluate_text_quality(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        score = 0.0
        length_score = min(1.0, len(text.strip()) / 1000)
        score += length_score * 0.3
        words = text.split()
        word_score = min(1.0, len(words) / 200)
        score += word_score * 0.2
        structure_indicators = ['experience', 'education', 'skills', 'summary']
        structure_score = sum(1 for indicator in structure_indicators
                                if indicator.lower() in text.lower()) / len(structure_indicators)
        score += structure_score * 0.3
        contact_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
        ]
        contact_score = sum(1 for pattern in contact_patterns 
                                if re.search(pattern, text)) / len(contact_patterns)
        score += contact_score * 0.2
        return score

    def _clean_extracted_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.replace('•', '•')  # Normalize bullet points
        text = text.replace('–', '-')  # Normalize dashes

        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+$', line) and len(line) <= 3:
                continue
            if len(line) < 5 and (not line or line.isdigit()):
                continue
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines).strip()

    def _detect_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        text_lower = text.lower()
        section_positions = {}
        for section_name, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(r'\b' + pattern + r'\b', text_lower))
                if matches:
                    section_positions[section_name] = matches[0].start()
                    break
        sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])
        for i, (section_name, start_pos) in enumerate(sorted_sections):
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][1]
            else:
                end_pos = len(text)
            section_content = text[start_pos:end_pos].strip()
            lines = section_content.split('\n')
            if lines:
                content_lines = lines[1:] if len(lines) > 1 else lines
                sections[section_name] = '\n'.join(content_lines).strip()
        return sections

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        contact_info = {}
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        phone_patterns = [
            r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                if isinstance(phones[0], tuple):
                    contact_info['phone'] = '-'.join(phones[0])
                else:
                    contact_info['phone'] = phones[0]
                break
        linkedin_patterns = [
            r'linkedin\.com/in/([A-Za-z0-9\-]+)',
            r'linkedin\.com/pub/([A-Za-z0-9\-]+)',
            r'/in/([A-Za-z0-9\-]+)'
        ]
        for pattern in linkedin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                contact_info['linkedin'] = matches[0]
                break
        github_patterns = [
            r'github\.com/([A-Za-z0-9\-]+)',
            r'github\.io/([A-Za-z0-9\-]+)'
        ]
        for pattern in github_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                contact_info['github'] = matches[0]
                break
        lines = text.split('\n')
        potential_names = []
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if not line or line.lower() in ['resume', 'curriculum vitae', 'cv']:
                continue
            words = line.split()
            if (2 <= len(words) <= 4 and 
                all(word.isalpha() and word[0].isupper() for word in words) and
                len(line) < 50):
                potential_names.append(line)
        if potential_names:
            contact_info['name'] = potential_names[0]
        address_patterns = [
            r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b',
            r'\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'
        ]
        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                contact_info['address'] = matches[0]
                break
        return contact_info

    def _analyze_document_structure(self, text: str, page_texts: List[str]) -> Dict[str, any]:
        analysis = {
            'estimated_layout': 'single_column',
            'section_count': 0,
            'average_line_length': 0,
            'text_density': 0,
            'formatting_quality': 0,
            'completeness_score': 0
        }
        if not text:
            return analysis
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            analysis['average_line_length'] = sum(len(line) for line in lines) / len(lines)
        line_lengths = [len(line) for line in lines if line]
        if line_lengths:
            length_variance = max(line_lengths) - min(line_lengths)
            if length_variance > 100:
                analysis['estimated_layout'] = 'two_column'
            elif length_variance > 200:
                analysis['estimated_layout'] = 'multi_section'
        sections_found = 0
        for section_name, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + pattern + r'\b', text.lower()):
                    sections_found += 1
                    break
        analysis['section_count'] = sections_found
        total_words = len(text.split())
        pages = len(page_texts)
        analysis['text_density'] = total_words / pages if pages > 0 else 0
        formatting_indicators = [
            r'•',
            r'\n\s*-',
            r'\d{4}\s*-\s*\d{4}',
            r'[A-Z][a-z]+\s+\d{4}',
        ]
        formatting_score = 0
        for pattern in formatting_indicators:
            if re.search(pattern, text):
                formatting_score += 1
        analysis['formatting_quality'] = formatting_score / len(formatting_indicators)
        essential_sections = ['experience', 'education', 'skills']
        completeness_score = 0
        for section in essential_sections:
            if any(re.search(r'\b' + pattern + r'\b', text.lower()) 
                   for pattern in self.section_patterns.get(section, [])):
                completeness_score += 1
        analysis['completeness_score'] = completeness_score / len(essential_sections)
        return analysis

    def _calculate_extraction_confidence(self, extraction_result: Dict) -> float:
        confidence_factors = []
        text_length = len(extraction_result['raw_text'])
        length_confidence = min(1.0, text_length / 2000)
        confidence_factors.append(length_confidence * 0.2)
        sections_found = len(extraction_result['sections'])
        section_confidence = min(1.0, sections_found / 5)
        confidence_factors.append(section_confidence * 0.3)
        contact_fields = len(extraction_result['contact_info'])
        contact_confidence = min(1.0, contact_fields / 4)
        confidence_factors.append(contact_confidence * 0.2)
        structure_analysis = extraction_result['structure_analysis']
        structure_confidence = (
            structure_analysis.get('formatting_quality', 0) * 0.5 +
            structure_analysis.get('completeness_score', 0) * 0.5
        )
        confidence_factors.append(structure_confidence * 0.3)
        return sum(confidence_factors)

    # ---- Other extraction/validation routines... ----

    def extract_work_experience(self, text: str) -> List[Dict[str, str]]:
        experiences = []
        experience_patterns = [
            r'([A-Za-z\s&]+)\s*[\-\|]\s*([A-Za-z\s,]+)\s*\n([A-Za-z]+\s+\d{4}\s*[-–]\s*(?:[A-Za-z]+\s+\d{4}|Present|Current))',
            r'([A-Za-z\s&]+)\s*,\s*([A-Za-z\s,]+)\s*\n([A-Za-z]+\s+\d{4}\s*[-–]\s*(?:[A-Za-z]+\s+\d{4}|Present|Current))',
        ]
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                experience = {
                    'title': match[0].strip(),
                    'company': match[1].strip(),
                    'duration': match[2].strip(),
                    'description': ''
                }
                experiences.append(experience)
        return experiences[:10]

    def extract_education(self, text: str) -> List[Dict[str, str]]:
        education_entries = []
        education_patterns = [
            r'(Bachelor|Master|PhD|Ph\.D|Doctorate|B\.S|B\.A|M\.S|M\.A|MBA)[\s\w]*in\s+([\w\s]+)\s*[-–]?\s*([A-Za-z\s,]+)\s*(?:,\s*)?(\d{4})?',
            r'([A-Za-z\s,]+University|[A-Za-z\s,]+College|[A-Za-z\s,]+Institute)\s*[-–]?\s*([\w\s]+)?\s*(\d{4})?'
        ]
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 3:
                    education = {
                        'degree': match[0].strip() if match[0] else '',
                        'field': match[1].strip() if len(match) > 1 and match[1] else '',
                        'institution': match[2].strip() if len(match) > 2 and match[2] else '',
                        'year': match[3].strip() if len(match) > 3 and match[3] else ''
                    }
                    education_entries.append(education)
        return education_entries[:5]

    def validate_pdf_content(self, pdf_file: Union[str, io.IOBase]) -> Dict[str, any]:
        validation_result = {
            'is_valid': False,
            'file_size': 0,
            'page_count': 0,
            'is_password_protected': False,
            'estimated_processing_time': 0,
            'warnings': []
        }
        try:
            pdf_data = self._load_pdf_binary(pdf_file)
            if pdf_data is None:
                raise ValueError("Could not read PDF as binary data.")

            validation_result['file_size'] = len(pdf_data)
            if validation_result['file_size'] > 10 * 1024 * 1024:
                validation_result['warnings'].append('Large file size may increase processing time')

            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            validation_result['page_count'] = pdf_document.page_count
            validation_result['is_password_protected'] = pdf_document.needs_pass
            validation_result['estimated_processing_time'] = pdf_document.page_count * 2  # est, seconds

            has_text = False
            for page_num in range(min(3, pdf_document.page_count)):
                page = pdf_document.load_page(page_num)
                if page.get_text().strip():
                    has_text = True
                    break
            if not has_text:
                validation_result['warnings'].append('PDF may be image-based and require OCR')

            pdf_document.close()
            validation_result['is_valid'] = True
        except Exception as e:
            validation_result['warnings'].append(f'PDF validation error: {str(e)}')
        return validation_result
