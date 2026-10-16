# CIFAR-10 Image Classifier — MLOps Pipeline

A production-ready MLOps project demonstrating end-to-end model lifecycle management:  
**DVC** for pipeline + data versioning → **S3** for artifact storage → **Docker** for packaging → **GitHub Actions** for CI/CD → **EC2** for serving.

## Architecture

```
GitHub Push
     │
     ▼
GitHub Actions
  ├── dvc pull (model from S3)
  ├── docker build → Docker Hub
  └── SSH → EC2 → docker run
```

## Project Structure

```
image-classifier/
├── .github/workflows/ci-cd.yaml   # CI/CD pipeline
├── src/
│   ├── prepare.py                 # Download & save CIFAR-10
│   ├── train.py                   # Train CNN
│   ├── evaluate.py                # Evaluate & write metrics.json
│   └── app.py                     # Flask REST API
├── templates/index.html           # Web UI
├── dvc.yaml                       # DVC pipeline stages
├── dvc.lock                       # Locked pipeline state (committed)
├── params.yaml                    # Hyperparameters
├── metrics.json                   # Evaluation results (committed)
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

## Quick Start (Local)

```bash
# 1. Create and activate venv
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run the DVC pipeline
dvc init
dvc repro

# 3. Check metrics
dvc metrics show
dvc dag

# 4. Run the app locally
cd image-classifier
gunicorn --bind 0.0.0.0:5000 src.app:app
# Visit http://localhost:5000
```

## DVC Pipeline Stages

| Stage | Command | Outputs |
|-------|---------|---------|
| `prepare` | `python src/prepare.py` | `data/processed/cifar10.npz` |
| `train` | `python src/train.py` | `models/model.keras` |
| `evaluate` | `python src/evaluate.py` | `metrics.json` |

## S3 Remote Setup

```bash
# Configure remote
dvc remote add -d s3remote s3://imgclf-dvc-<yourname>/dvcstore
dvc remote modify s3remote region ap-south-1

# Push artifacts
dvc push
```

## Docker

```bash
# Build
docker build -t image-classifier .

# Run
docker run -p 5000:5000 image-classifier
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check → `{"status":"ok"}` |
| `POST` | `/predict` | `multipart/form-data` with `file` field → JSON prediction |

### Example

```bash
curl -X POST http://<EC2_IP>:5000/predict \
  -F "file=@cat.jpg"
# {"label":"cat","confidence":0.823,"top3":[...]}
```

## GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM access key |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key |
| `AWS_REGION` | e.g. `ap-south-1` |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `EC2_HOST` | EC2 public / Elastic IP |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | Full contents of `.pem` key |

## Triggering a Retrain

```bash
# Edit params.yaml (e.g. increase epochs)
dvc repro
dvc push
git add dvc.lock params.yaml metrics.json
git commit -m "Retrain: epochs=10"
git push
# → CI/CD automatically redeploys the new model
```

## Live Demo

🌐 `http://<EC2_ELASTIC_IP>:5000`

## CIFAR-10 Classes

✈️ airplane · 🚗 automobile · 🐦 bird · 🐱 cat · 🦌 deer  
🐶 dog · 🐸 frog · 🐴 horse · 🚢 ship · 🚚 truck
