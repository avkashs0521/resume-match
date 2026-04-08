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
    # Convert list/dict to clean strings for the log parser
    xai_str = json.dumps(xai).replace(' ', '') if xai else "null"
    print_log(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} xai={xai_str} error={error}")

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print_log(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")

def run_inference():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_log("[ERROR] OPENAI_API_KEY not found in environment.")
        return

    client = OpenAI(api_key=api_key)
    tasks = ["easy", "medium", "hard"]
    
    for task_name in tasks:
        log_start(task_name)
        env = ResumeEnv(task_type=task_name)
        obs = env.reset()
        done = False
        
        rewards = []
        steps_taken = 0
        error_msg = "null"
        success = False
        score = 0.0

        try:
            while not done:
                steps_taken += 1
                
                # Setup structured prompt
                resumes_str = json.dumps([r.model_dump() for r in obs.resumes])
                jobs_str = json.dumps([j.model_dump() for j in obs.jobs])
                
                prompt = (
                    f"Task Type: {task_name}\n"
                    f"Goal: Match jobs to resumes optimally.\n"
                    f"Jobs Available: {jobs_str}\n"
                    f"Resumes Available: {resumes_str}\n"
                )
                
                if task_name == "easy":
                    prompt += "Match the single job ID to the best resume ID. Set Action.matches={'job_id': 'resume_id'}."
                elif task_name == "medium":
                    prompt += "Rank the top 3 resume IDs for the job. Set Action.ranked_list=['r1', 'r2', 'r3']."
                else:
                    prompt += "Assign all jobs to distinct resumes. Set Action.matches={'j1': 'r1', 'j2': 'r2', ...}."

                try:
                    res = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an AI HR Agent specialized in OpenEnv simulations."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format=Action,
                    )
                    action = res.choices[0].message.parsed
                except Exception as ex:
                    error_msg = str(ex).replace('\n', ' ')
                    action = Action() # Null action fallback
                
                obs, reward, done, info = env.step(action)
                
                # Format action string for log parser
                if task_name == "medium": 
                    act_str = str(action.ranked_list).replace(' ', '')
                else: 
                    act_str = str(action.matches).replace(' ', '')
                
                # Capture XAI metadata for logs
                xai_metadata = {
                    "matched": reward.matched_skills,
                    "missing": reward.missing_skills,
                    "suggestion": reward.suggestion
                }

                rewards.append(reward.score)
                log_step(steps_taken, act_str, reward.score, done, xai=xai_metadata, error=error_msg)
                
            score = rewards[-1] if rewards else 0.0
            success = score > 0.3 # Threshold for "Success"
        except Exception as e:
            error_msg = str(e).replace('\n', ' ')
            log_step(steps_taken, "error", 0.0, True, error="internal_crash")
            score = 0.0
            success = False
            
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    run_inference()