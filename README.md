## Summary

- Project: Vehicle detection and tracking (YOLO-based)
- Local path: `c:\Users\ASUS\Desktop\PROJE`

# AIPoweredCarTracking

This repository contains code and resources for a quick prototype of an AI-powered car tracking project based on YOLO-style object detection models.

## Overview

- Project: Vehicle detection and tracking using YOLO-based models
- Local workspace: `c:\Users\ASUS\Desktop\PROJE`

## Quick start

1. Clone the repository:

```powershell
git clone https://github.com/YunusKok/AIPoweredCarTracking.git
cd AIPoweredCarTracking
```

2. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the main script (example):

```powershell
python main_car.py
```

## Model weights (important)

- This project may include model weight files such as `*.pt` or `*.pth` (for example `yolov8n.pt`).
- Model weight files are often large. Consider repository size limits and single-file size limits on GitHub.
- Recommendation: Do not add large weights directly to the repository. Use Git LFS or host weights on cloud storage (Google Drive, S3, etc.) and download them at runtime.

Example Git LFS usage:

```powershell
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add path\to\model.pt
git commit -m "Add model tracked by LFS"
git push
```

## Contributing

- Use feature branches like `feature/<username>/<short-description>` (e.g. `feature/yunus/object-tracking`).
- Push changes and create a Pull Request on GitHub. Request at least one review before merging.
- We recommend protecting the `main` branch and requiring PR reviews before merge.

## Repository structure

- `main_car.py` — main entry point for running the demo/experiment.
- `requirements.txt` — Python dependencies.
- `yolov8n.pt` — (optional) model weight file. Use Git LFS if you store it in the repo.

## Contact / Support

- If you have questions or want to contribute, please open an issue on the repository or contact `@YunusKok`.

## License

No license is specified for this project. If you'd like to add one, add a `LICENSE` file (e.g. MIT).

## Notes

- Replace `<username>` and `<repo-name>` in the examples above with your GitHub username and repository name.

- Create feature branches using the pattern `feature/<username>/<short-description>` (e.g. `feature/yunus/object-tracking`).
- Open a Pull Request after pushing your changes and request at least one review.
- Prefer merging via PRs instead of pushing directly to `main`. Consider enabling branch protection rules for `main` (Require pull request reviews).

## Project Structure

- `main_car.py` — main entry point for running demos or experiments.
- `requirements.txt` — Python dependencies.
- `yolov8n.pt` — (optional) model weights; large file — consider LFS.

## Contact

- For questions or contributions, please open an Issue in this repository or contact `@YunusKok` on GitHub.

## License

No license is specified for this project.

## Notes

- Replace `<username>` and `<repo-name>` in the commands above with your GitHub username and repository name.
