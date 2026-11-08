# Intelligent ML-Based Resume Screening & ATS Decision-Support System
> **Information Systems Engineering (ISE) Final Capstone Project**  
> An Explainable AI (XAI) Application Powered by NLP Vector Space Modeling, Sublinear TF-IDF, Multi-Criteria Decision Analysis (MCDA), and Ethical Blind Screening.

---

## 1. Executive Summary & Motivation
In modern recruitment engineering, enterprise talent acquisition teams receive hundreds to thousands of applicant resumes for a single job opening. Traditional Applicant Tracking Systems (ATS) often rely on crude, opaque keyword counting that is vulnerable to "keyword stuffing" or penalizes qualified non-traditional candidates. Conversely, purely black-box deep learning models trained on historical hiring data have notoriously reproduced and amplified historical demographic biases (most famously documented in Amazon's 2018 scrapped hiring tool).

This project presents an **Explainable, Transparent, and Robust Machine Learning Decision-Support System** designed to assist human recruiters. It combines:
1. **Unsupervised NLP Vector Space Modeling** (TF-IDF with Sublinear Term Frequency and unigram/bigram n-grams).
2. **High-Dimensional Cosine Similarity** for semantic alignment independent of document length.
3. **Multi-Domain Skill Taxonomy NER** spanning 350+ categorized technical and soft skills.
4. **Multi-Criteria Decision Analysis (MCDA)** scoring that integrates skill coverage, semantic similarity, verified years of experience, and highest education attainment.
5. **Ethical AI & Blind Screening** compliance (NYC Local Law 144 and EU AI Act alignment) to strip personally identifiable information (PII) during initial shortlisting.

---

## 2. Key Features

### Dual-Workflow Operation
- **Instant Demo Benchmarking:** Includes 5 pre-configured industry job descriptions (*Senior ML Engineer, Full Stack Web Developer, Data Analyst, Cloud & DevOps Engineer, Cybersecurity Analyst*) and 5 realistic multi-tiered applicant resumes for instantaneous demonstration during evaluations or vivas without needing external files.
- **Real-World File Ingestion:** Seamlessly parse and screen uploaded resumes across **PDF**, **DOCX**, **TXT**, and bulk **ZIP archives**.

### Multi-Factor Scoring Engine (MCDA)
Unlike simplistic keyword counters, the overall candidate relevance score is a customizable weighted composite:
- **Technical Skill Match (45% default):** Proportion of required technical skills present in the resume.
- **Semantic Text Similarity (35% default):** Cosine similarity between the TF-IDF representation of the job description and the resume text.
- **Experience Alignment (10% default):** Automated heuristic extraction of verified years of experience compared against job requirements.
- **Education Attainment (10% default):** Degree level evaluation (Ph.D., Master's, Bachelor's, Associate's).
- *All weights are interactively adjustable via sidebar controls in real time.*

### Deep-Dive Candidate Gap Analysis & Recruiter Tools
- Individual candidate score gauges.
- **Skill Gap Matrix:** Color-coded badges for *Matched Skills*, *Missing Skills* (actionable gap analysis), and *Bonus Skills*.
- **Domain Competency Radar Chart:** Interactive spider chart comparing job demands against candidate strengths across 9 engineering domains.
- **Head-to-Head Comparison Tool:** Direct grouped bar comparison and skill Venn differential between two finalist candidates.
- **Technical Interview Question Generator:** Auto-generates role-specific questions targeted at verified skills and probes identified skill gaps, with interviewer rubrics.
- Direct extraction of contact information: Email, Phone, clickable LinkedIn, GitHub, and Portfolio URLs.

### Real-Time ATS Resume Auditor & Optimizer
- An integrated tool for candidates and job seekers to audit their resume's ATS readiness.
- Evaluates word count, bullet point density, section completeness (Summary, Experience, Education, Skills, Projects, Certifications).
- Generates actionable recommendations to optimize ATS parsing.

### Ethical AI & Blind Screening Mode
- With a single toggle, candidate names, emails, phones, and social links are masked.
- Ensures initial shortlisting is objective and merit-based, eliminating unconscious demographic bias.

### Multi-Format Export Engine
- Export leaderboard data to CSV.
- Download structured executive summary shortlists as `.txt`.
- Download printable executive candidate scorecards as standalone `.html` reports ready to print or save as PDF.

---

## 3. Software Engineering Architecture

```
                                  +---------------------------+
                                  |   Job Description Input   |
                                  |   (Pasted Text or Sample) |
                                  +-------------+-------------+
                                                |
+-------------------------------+               |
| Candidate Resumes (.pdf/.docx)|               |
+---------------+---------------+               |
                |                               |
                v                               v
+-------------------------------+ +---------------------------+
| Multi-Format Ingestion Engine | | Regex Skill Extraction    |
| (pypdf, python-docx, io.Bytes)| | (350+ Categorized Taxonomy|
+---------------+---------------+ +-------------+-------------+
                |                               |
                v                               |
+-------------------------------+               |
| Text Preprocessing & Cleaning |               |
| (Normalize C++, C#, .NET, etc)|               |
+---------------+---------------+               |
                |                               |
                v                               |
+-------------------------------+               |
|   TF-IDF Vector Space Model   |               |
| (ngram=(1,2), sublinear_tf)   |               |
+---------------+---------------+               |
                |                               |
                v                               |
+-------------------------------+               |
| Cosine Similarity Computation |               |
+---------------+---------------+               |
                |                               |
                +---------------+---------------+
                                |
                                v
                +-------------------------------+
                | Multi-Criteria Decision (MCDA)|
                | Skills + Semantic + Exp + Edu |
                +---------------+---------------+
                                |
                                v
                +-------------------------------+
                |   Streamlit Web Application   |
                |  (Leaderboard, Radar, Export) |
                +-------------------------------+
```

---

## 4. Mathematical & Algorithmic Foundations

### Sublinear TF-IDF Vector Space Model
To neutralize "keyword stuffing" (where a candidate repeats words to game search algorithms), sublinear term frequency scaling is applied:
$$\text{TF}(t, d) = \begin{cases} 1 + \ln(\text{count}(t, d)) & \text{if } \text{count}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

Inverse Document Frequency (IDF) scales down ubiquitous terms:
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### High-Dimensional Cosine Similarity
Calculates the angular similarity between the Job Description vector $\vec{J}$ and candidate resume vector $\vec{R}$ in the $N$-dimensional unigram/bigram feature space:
$$\text{Cosine Similarity}(\vec{J}, \vec{R}) = \frac{\vec{J} \cdot \vec{R}}{\|\vec{J}\| \|\vec{R}\|} = \frac{\sum_{i=1}^{n} J_i R_i}{\sqrt{\sum_{i=1}^{n} J_i^2} \sqrt{\sum_{i=1}^{n} R_i^2}}$$

### Multi-Criteria Composite Score (MCDA)
$$\text{Composite Score} = w_{\text{skill}} \cdot S_{\text{skill}} + w_{\text{semantic}} \cdot S_{\text{semantic}} + w_{\text{exp}} \cdot S_{\text{exp}} + w_{\text{edu}} \cdot S_{\text{edu}}$$
$$\text{where } \sum w_i = 1.0$$

---

## 5. Project Directory Structure
```
ML_ISE/
|-- app.py                      # Main Streamlit application and multi-tab interface
|-- requirements.txt            # System dependencies
|-- README.md                   # Comprehensive academic and practical documentation
|-- run_app.bat                 # Windows 1-click launcher script
|-- sample_resumes/             # Pre-built benchmark test resumes (.docx, .txt)
`-- modules/
    |-- __init__.py             # Python package marker
    |-- skills_db.py            # 350+ Categorized skills taxonomy across 9 domains
    |-- extractor.py            # PDF/DOCX/TXT text parsing, entity & contact extraction, ATS auditor
    |-- matcher.py              # TF-IDF, Cosine Similarity, MCDA decision engine
    |-- sample_data.py          # Benchmark job descriptions and realistic candidate resumes
    `-- visualizations.py       # Plotly charts (Leaderboard, Radar chart, Score gauges)
```

---

## 6. Setup & Installation Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.

### 1. Clone or Extract the Project
Open a terminal in the project directory:
```bash
cd c:\Users\Sajjad\Downloads\ML_ISE
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
python -m streamlit run app.py
```
Or double-click `run_app.bat` on Windows.
The application opens at `http://localhost:8501`.

---

## 7. Viva Voce & Academic Defense Q&A Cheat Sheet

| Question | Examiner's Focus | Recommended Answer |
| :--- | :--- | :--- |
| **Q1: Why not train an end-to-end Neural Classifier like a BERT classification head?** | Supervised vs. Unsupervised Machine Learning | Supervised hiring classifiers require tens of thousands of labeled historical hiring decisions. Such datasets reflect past human hiring biases (e.g. Amazon AI scandal). Furthermore, an unsupervised vector space model combined with explicit skill taxonomy extraction is fully transparent, explainable (XAI), and dynamically adapts to any newly posted job description without requiring retraining. |
| **Q2: How does the system prevent keyword manipulation?** | Robustness & Adversarial Inputs | We use **sublinear TF scaling** ($1 + \ln(tf)$). Repeating a keyword 30 times yields negligible incremental score. In addition, skill matching checks for distinct required skills rather than repetitive counts. |
| **Q3: What makes this compliant with AI Ethics regulations?** | Algorithmic Fairness & Legal Compliance | The system provides an interactive **Blind Screening Mode** that masks candidate names, emails, phone numbers, and demographics. Every score is decomposed into explainable sub-factors, allowing human recruiters to audit decisions in accordance with NYC Local Law 144 and the EU AI Act. |
| **Q4: How does the system handle complex technical terms like C++, C#, and .NET?** | NLP Tokenization & Regex Boundaries | Standard tokenizers strip punctuation, reducing `C++` to `C` and `.NET` to `NET`. Our custom normalization engine uses boundary-safe token lookarounds and explicit preservation rules for tech symbols. |

---

## 8. License & Academic Integrity
Developed as an academic capstone project for Information Systems Engineering (ISE). Built with open-source Python libraries (`Streamlit`, `Scikit-Learn`, `Pandas`, `Plotly`, `PyPDF`, `python-docx`).
