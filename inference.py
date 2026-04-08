import os
import sys
import json
from openai import OpenAI
from app.env.environment import ResumeEnv
from app.env.models import Action

import logging
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

old_stdout = sys.stdout
sys.stdout = sys.stderr

def print_log(msg):
    old_stdout.write(msg + "\n")
    old_stdout.flush()

def log_start(task):
    print_log(f"[START] task={task} env=resume-matching-env model=gpt-4o-mini")

def log_step(step, action, reward, done, xai=None, error="null"):
    xai_str = json.dumps(xai).replace(' ', '') if xai else "null"
    print_log(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} xai={xai_str} error={error}")

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print_log(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")

def run_inference():
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        print_log("[ERROR] HF_TOKEN not found in environment.")
        return

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )

    tasks = ["easy", "medium", "hard"]

    for task_name in tasks:
        log_start(task_name)
        env = ResumeEnv(task_type=task_name)
        obs = env.reset()

        done = False
        rewards = []
        steps_taken = 0
        success = False
        score = 0.0
        prev_feedback = "None"

        try:
            while not done:
                steps_taken += 1
                error_msg = "null"

                # -------------------------------
                # 🔥 BUILD CLEAN CONTEXT
                # -------------------------------
                resume_text = ""
                for r in obs.resumes:
                    resume_text += f"{r.id}: {r.text[:120]}\n"

                job_text = ""
                for j in obs.jobs:
                    job_text += f"{j.id}: {j.description} | Skills: {j.skills_required}\n"

                # -------------------------------
                # 🔥 PROMPT
                # -------------------------------
                job_ids = [j.id for j in obs.jobs]
                resume_ids = [r.id for r in obs.resumes]

                resume_text = "\n".join([
                    f"{r.id}: {r.text[:100]}" for r in obs.resumes
                ])

                job_text = "\n".join([
                    f"{j.id}: {j.description} | Skills: {j.skills_required}" for j in obs.jobs
                ])

                prompt = f"""
You are an AI HR agent.

Task: {task_name}

VALID JOB IDS:
{job_ids}

VALID RESUME IDS:
{resume_ids}

Jobs:
{job_text}

Resumes:
{resume_text}

STRICT RULES:
- ONLY use job IDs from: {job_ids}
- ONLY use resume IDs from: {resume_ids}
- DO NOT invent IDs
- Match based on skill overlap

"""
                if task_name == "easy":
                    prompt += """
Return:
{ "matches": { "job_id": "resume_id" } }
"""
                elif task_name == "medium":
                    prompt += """
Return:
{ "ranked_list": ["r1","r2","r3"] }
"""
                else:
                    prompt += """
Return:
{ "matches": { "j0": "r1", "j1": "r2" } }
"""

                # -------------------------------
                # 🔥 FIXED SCHEMA
                # -------------------------------
                action_schema = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ActionResponse",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "matches": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        # allow at least one known key (dynamic workaround)
                                        **{j.id: {"type": "string"} for j in obs.jobs}
                                    },
                                    "required": [j.id for j in obs.jobs]
                                },
                                "ranked_list": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "additionalProperties": False,
                            "required": ["matches", "ranked_list"]
                        }
                    }
                }

                try:
                    res = client.chat.completions.create(
                        model="openai/gpt-oss-120b:groq",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an AI HR Agent specialized in resume-job matching."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        response_format=action_schema,
                    )

                    parsed_json = json.loads(res.choices[0].message.content)

                    action = Action(
                        matches=parsed_json.get("matches", {}),
                        ranked_list=parsed_json.get("ranked_list", [])
                    )

                except Exception as ex:
                    error_msg = str(ex).replace('\n', ' ')
                    action = Action()

                # -------------------------------
                # STEP ENV
                # -------------------------------
                obs, reward, done, info = env.step(action)

                # Format action string with human-readable titles for logs
                if task_name == "medium":
                    translated = []
                    for r_id in action.ranked_list:
                        r_obj = next((r for r in obs.resumes if r.id == r_id), None)
                        skill_preview = ",".join(r_obj.skills[:2]) if r_obj else ""
                        translated.append(f"[{r_id}] {skill_preview}")
                    act_str = str(translated).replace("'", "")
                else:
                    translated = {}
                    for j_id, r_id in action.matches.items():
                        j_obj = next((j for j in obs.jobs if j.id == j_id), None)
                        r_obj = next((r for r in obs.resumes if r.id == r_id), None)
                        
                        j_title = j_obj.description.split("skilled")[0].replace("Seeking a ", "").replace("Seeking an ", "").strip() if j_obj else j_id
                        r_preview = f"[{r_id}] {','.join(r_obj.skills[:2])}" if r_obj else r_id
                        translated[j_title] = r_preview
                    act_str = json.dumps(translated)

                xai_metadata = {
                    "matched": reward.matched_skills,
                    "missing": reward.missing_skills,
                    "suggestion": reward.suggestion
                }

                prev_feedback = json.dumps(xai_metadata)

                rewards.append(reward.score)

                log_step(
                    steps_taken,
                    act_str,
                    reward.score,
                    done,
                    xai=xai_metadata,
                    error=error_msg
                )

            score = rewards[-1] if rewards else 0.0
            success = score > 0.3

        except Exception:
            log_step(steps_taken, "error", 0.0, True, error="internal_crash")
            score = 0.0
            success = False

        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    run_inference()