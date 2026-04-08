# 🏢 OpenEnv-Compliant Resume-Job Matching Environment

A strictly OpenEnv-compliant real-world simulation environment for Job and Resume Matching.

## Environment Description 🧠
The `resume-matching-env` simulates the authentic task of an HR AI assistant matching candidate resumes to job descriptions. 
Unlike standard vector-search, this environment utilizes a **Hybrid NLP weighting strategy** ($90/10$ split between TF-IDF and SentenceTransformers) to deterministically assess and reward the AI agent's actions natively without bias.

## Action & Observation Spaces

### Observation Space
The observation space is a Pydantic model (`Observation`) containing:
- `resumes`: A list of resumes (id, skills, experience, text).
- `jobs`: A list of job requirements (id, skills_required, description).
- `current_matches`: The current state of assignments.
- `step_count`: The number of steps taken in the current episode.

### Action Space
The action space is a Pydantic model (`Action`) containing:
- `matches`: A dictionary of `{job_id: resume_id}` for direct assignment tasks.
- `ranked_list`: A list of `resume_id`s corresponding to the top-ranked candidates for a specific job.

## Tasks and Difficulty
The environment provides 3 tasks:
- **Easy (`easy`)**: The agent is presented with 1 job and must identify the single best matching resume.
- **Medium (`medium`)**: The agent is presented with 1 job and must return a ranked list of the top 3 resumes.
- **Hard (`hard`)**: The agent is presented with a batch of 5 jobs and must optimally assign each job to a single resume. The agent can submit assignments iteratively over multiple steps.

## Setup Instructions
```bash
pip install -r requirements.txt
```

## Running the External Agent Interface (Inference)
You can evaluate a standard LLM agent's performance against the environment using the inference pipeline. This simulates the Phase 2 agent-runner that will interact with the environment.
```bash
export OPENAI_API_KEY="sk-..."
python inference.py
```

## Evaluation (Phase 2) Compatibility 🤖
This setup strictly complies with the Phase 2 OpenEnv requirements:
- **No internal RL agent required:** The core deliverable is the Environment + Inference bridge.
- **`inference.py` script:** Facilitates interaction between an external LLM agent (like OpenAI) and the `ResumeEnv` environment.
- **Structured JSON-compliant logs:** Built to output `[START]`, `[STEP]`, and `[END]` syntax tracking the state, rewards, and exact agent XAI feedback, ensuring seamless ingestion by automated benchmarking scrapers.

## Deployment
This environment is containerized for Hugging Face Spaces.
```bash
docker build -t openenv-resume .
docker run -p 7860:7860 openenv-resume
```
