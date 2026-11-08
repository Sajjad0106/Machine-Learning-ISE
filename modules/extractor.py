"""
Text and Entity Extraction Module for Resumes.
Extracts raw text from PDF, DOCX, and TXT files, and identifies key candidate metadata
including contact information, education, experience, and ATS resume sections.
"""

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


def extract_text_from_file(uploaded_file) -> str:
    """Extract raw text from an uploaded file object or Path."""
    name = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else str(uploaded_file).lower()
    
    if hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        raw_bytes = uploaded_file.read()
    else:
        with open(uploaded_file, "rb") as f:
            raw_bytes = f.read()

    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is required to read PDF files.")
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError("No selectable text found. This may be a scanned or image-only PDF.")
        return text

    if name.endswith(".docx"):
        if Document is None:
            raise RuntimeError("python-docx is required to read DOCX files.")
        doc = Document(io.BytesIO(raw_bytes))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    parts.append(row_text)
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("The DOCX document contains no readable text.")
        return text

    if name.endswith(".txt"):
        for encoding in ["utf-8", "utf-16", "latin-1"]:
            try:
                return raw_bytes.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace").strip()

    raise ValueError(f"Unsupported format '{Path(name).suffix}'. Please upload PDF, DOCX, TXT, or ZIP.")


def extract_resumes_from_zip(zip_bytes: bytes) -> Dict[str, str]:
    """Extract all valid PDF, DOCX, and TXT resumes from a ZIP archive."""
    resumes = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for file_info in z.infolist():
            if file_info.is_dir() or file_info.filename.startswith("__MACOSX"):
                continue
            fname = Path(file_info.filename).name
            fname_lower = fname.lower()
            if fname_lower.endswith((".pdf", ".docx", ".txt")):
                file_bytes = z.read(file_info.filename)
                
                class _VirtualFile:
                    def __init__(self, name, data):
                        self.name = name
                        self.data = data
                    def getvalue(self):
                        return self.data
                        
                try:
                    txt = extract_text_from_file(_VirtualFile(fname, file_bytes))
                    if txt.strip():
                        resumes[fname] = txt
                except Exception:
                    continue
    return resumes


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """Extract email, phone, LinkedIn, and GitHub links from resume text."""
    # Email regex
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = re.findall(email_pattern, text)
    email = emails[0] if emails else None

    # Phone regex (international & domestic formats)
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{3,5}'
    phones = re.findall(phone_pattern, text)
    valid_phone = None
    for p in phones:
        clean = re.sub(r'\D', '', p)
        if 8 <= len(clean) <= 15:
            valid_phone = p.strip()
            break

    # Social and repo links
    linkedin_match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    linkedin = f"linkedin.com/in/{linkedin_match.group(1)}" if linkedin_match else None

    github_match = re.search(r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    github = f"github.com/{github_match.group(1)}" if github_match else None

    portfolio_match = re.search(r'(?:https?://)?(?:[a-zA-Z0-9-]+\.)+(?:com|org|io|dev|me|net)/?(?:[a-zA-Z0-9-_/]+)?', text, re.IGNORECASE)
    portfolio = None
    if portfolio_match:
        url = portfolio_match.group(0)
        if not any(domain in url.lower() for domain in ["linkedin", "github", "gmail", "yahoo", "outlook", "hotmail"]):
            portfolio = url

    return {
        "email": email,
        "phone": valid_phone,
        "linkedin": linkedin,
        "github": github,
        "portfolio": portfolio
    }


def extract_candidate_name(text: str, filename: str) -> str:
    """Heuristic extraction of candidate name from the top of the resume."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    stop_words = {"resume", "curriculum", "vitae", "cv", "page", "contact", "profile", "summary", "experience"}
    
    for line in lines[:5]:
        clean = re.sub(r'[^a-zA-Z\s]', '', line).strip()
        words = clean.split()
        if 2 <= len(words) <= 4:
            if not any(w.lower() in stop_words for w in words):
                # Ensure it's not all lowercase and looks like a name
                if all(w[0].isupper() for w in words if w):
                    return clean
                elif clean.isupper():
                    return clean.title()

    # Fallback to filename without extension
    clean_filename = Path(filename).stem
    clean_filename = re.sub(r'[_.-]', ' ', clean_filename)
    clean_filename = re.sub(r'(?i)\b(resume|cv|updated|profile|final)\b', '', clean_filename).strip()
    return clean_filename.title() if clean_filename else Path(filename).stem


def extract_education_level(text: str) -> Dict[str, Any]:
    """Detect candidate's highest level of education and degree mentions."""
    text_lower = text.lower()
    
    levels = {
        "PhD / Doctorate": [r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctoral\b"],
        "Master's Degree": [r"\bmaster(?:'s)?\b", r"\bm\.?s\.?\b", r"\bm\.?tech\b", r"\bm\.?sc\b", r"\bmba\b", r"\bmca\b"],
        "Bachelor's Degree": [r"\bbachelor(?:'s)?\b", r"\bb\.?s\.?\b", r"\bb\.?tech\b", r"\bb\.?e\.?\b", r"\bb\.?sc\b", r"\bbca\b", r"\bundergraduate\b"],
        "Associate's / Diploma": [r"\bassociate(?:'s)?\b", r"\bdiploma\b", r"\bpolytechnic\b"]
    }

    detected_degrees = []
    highest_level = "Not Specified"

    for level, patterns in levels.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                detected_degrees.append(level)
                if highest_level == "Not Specified":
                    highest_level = level
                break

    # Common majors
    majors = [
        "computer science", "software engineering", "information technology",
        "data science", "artificial intelligence", "electrical engineering",
        "mechanical engineering", "cybersecurity", "mathematics", "statistics",
        "physics", "business administration", "information systems"
    ]
    detected_majors = [m.title() for m in majors if m in text_lower]

    return {
        "highest_level": highest_level,
        "all_levels": list(set(detected_degrees)),
        "majors": detected_majors
    }


def extract_experience_years(text: str) -> float:
    """Estimate total years of professional experience using explicit statements and date spans."""
    # Pattern 1: Explicit statements like "5+ years of experience" or "3 years exp"
    explicit_matches = re.findall(r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience', text, re.IGNORECASE)
    if explicit_matches:
        try:
            return max(float(m) for m in explicit_matches)
        except ValueError:
            pass

    # Pattern 2: Year intervals like 2018 - 2023, 2020 - Present
    years = re.findall(r'\b(20[0-2][0-9]|199[0-9])\s*(?:-|–|to)\s*(20[0-2][0-9]|present|current)\b', text, re.IGNORECASE)
    total_months = 0
    current_year = 2026

    for start_yr, end_yr in years:
        try:
            s = int(start_yr)
            e = current_year if end_yr.lower() in ["present", "current"] else int(end_yr)
            diff = max(0, e - s)
            if 0 < diff <= 30:
                total_months += diff * 12
        except ValueError:
            continue

    if total_months > 0:
        # Cap realistic experience calculation to prevent overlapping overlaps
        estimated_years = min(round(total_months / 12, 1), 25.0)
        return estimated_years

    return 0.0


def audit_resume_structure(text: str) -> Dict[str, Any]:
    """Audit the structural completeness and quality of the resume for ATS readability."""
    text_lower = text.lower()
    
    section_patterns = {
        "Contact Information": [r"email", r"phone", r"github", r"linkedin", r"contact"],
        "Professional Summary": [r"summary", r"objective", r"profile", r"about me"],
        "Work Experience": [r"experience", r"employment", r"work history", r"career"],
        "Education": [r"education", r"academic", r"degree", r"university", r"college"],
        "Skills": [r"skills", r"technical proficiencies", r"core competencies", r"technologies"],
        "Projects": [r"projects", r"personal projects", r"key initiatives"],
        "Certifications": [r"certifications", r"licenses", r"courses", r"accreditations"]
    }

    found_sections = []
    missing_sections = []

    for section, patterns in section_patterns.items():
        if any(re.search(rf"\b{pat}\b", text_lower) for pat in patterns):
            found_sections.append(section)
        else:
            missing_sections.append(section)

    word_count = len(text.split())
    bullet_count = len(re.findall(r'^\s*[-•*]\s+', text, re.MULTILINE))
    has_metrics = bool(re.search(r'\b\d+(?:%|\+|x|k|m)?\b', text))

    # Calculate ATS Health Score (0 - 100)
    score = 0
    # Sections check: up to 50 pts
    score += (len(found_sections) / len(section_patterns)) * 50
    # Length check: up to 25 pts (ideal: 400 - 1000 words)
    if 400 <= word_count <= 1200:
        score += 25
    elif 250 <= word_count < 400 or 1200 < word_count <= 1800:
        score += 15
    else:
        score += 5
    # Action metrics & bullet points: up to 25 pts
    if bullet_count >= 5:
        score += 15
    elif bullet_count >= 1:
        score += 8
    if has_metrics:
        score += 10

    suggestions = []
    if "Professional Summary" in missing_sections:
        suggestions.append("Add a concise 3-4 sentence professional summary at the top.")
    if "Projects" in missing_sections:
        suggestions.append("Include a dedicated Projects section showcasing practical system implementations.")
    if bullet_count < 5:
        suggestions.append("Use bullet points rather than dense paragraphs under experience.")
    if not has_metrics:
        suggestions.append("Quantify your achievements using metrics (e.g., 'improved latency by 35%').")
    if word_count < 300:
        suggestions.append("Resume content is sparse. Elaborate on project deliverables and technical stacks.")

    return {
        "ats_health_score": round(score, 1),
        "word_count": word_count,
        "bullet_count": bullet_count,
        "found_sections": found_sections,
        "missing_sections": missing_sections,
        "suggestions": suggestions
    }

