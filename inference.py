import os
from app.env.environment import ResumeEnv
from app.env.models import Action
from app.matching.matcher import match_medium
# 🔥 REQUIRED ENV VARIABLES
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔥 suppress transformers logs
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
API_BASE_URL = os.getenv("API_BASE_URL", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "rule-based")
HF_TOKEN = os.getenv("HF_TOKEN", "none")

TASK_NAME = "medium"
BENCHMARK = "resume_env"


def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


def run():
    env = ResumeEnv(task_type=TASK_NAME)

    rewards = []
    steps_taken = 0

    log_start()

    try:
        obs = env.reset()

        resumes = [r.model_dump() for r in obs.resumes]
        jobs = [j.model_dump() for j in obs.jobs]

        # ✅ use real matching logic
        ranked = match_medium(resumes, jobs)

        action = Action(ranked_list=ranked)

        obs, reward, done, _ = env.step(action)

        reward_val = reward.score if hasattr(reward, "score") else 0.0

        rewards.append(reward_val)
        steps_taken = 1

        log_step(1, str(ranked), reward_val, done)

        score = max(0.0, min(1.0, reward_val))
        success = score >= 0.1

    except Exception as e:
        log_step(1, "error", 0.0, True)
        score = 0.0
        success = False

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    run()