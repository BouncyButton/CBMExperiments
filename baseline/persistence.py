#!/usr/bin/env python3
import argparse
import os
import sys


def _get_wandb_api_key():
    # Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret("WANDB_API_KEY")
    except Exception:
        pass

    # Colab secrets
    try:
        from google.colab import userdata
        return userdata.get('WANDB_API_KEY')
    except Exception:
        pass

    # Env var fallback
    return os.environ.get('WANDB_API_KEY')


def _write_netrc(api_key, netrc_path):
    netrc_content = f"""machine api.wandb.ai
  login user
  password {api_key}
"""
    with open(netrc_path, "w") as f:
        f.write(netrc_content)
    os.chmod(netrc_path, 0o600)


def ensure_wandb_login(netrc_path):
    api_key = _get_wandb_api_key()
    if not api_key:
        raise RuntimeError("WANDB_API_KEY not found. Set it in Kaggle/Colab secrets or env var.")
    _write_netrc(api_key, netrc_path)


def log_model_artifact(model_path, artifact_name, artifact_type, project, entity, alias, netrc_path):
    import wandb
    ensure_wandb_login(netrc_path)

    run = wandb.init(project=project, entity=entity, job_type="log_model", reinit=True)
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(model_path)
    run.log_artifact(artifact, aliases=[alias] if alias else None)
    run.finish()


def download_model_artifact(artifact_path, output_dir, filename, netrc_path):
    import wandb
    ensure_wandb_login(netrc_path)

    run = wandb.init(project=None, entity=None, job_type="download_model", reinit=True)
    artifact = wandb.use_artifact(artifact_path, type="model")
    artifact_dir = artifact.download(root=output_dir)

    if filename:
        src = os.path.join(artifact_dir, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Expected file not found in artifact: {src}")
        print(src)
    else:
        print(artifact_dir)
    run.finish()


def main():
    parser = argparse.ArgumentParser(description="W&B model artifact persistence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    log_p = subparsers.add_parser("log", help="Log a model file as a W&B artifact")
    log_p.add_argument("--model_path", required=True, help="Path to model file (e.g., .pth)")
    log_p.add_argument("--artifact_name", required=True, help="Artifact name")
    log_p.add_argument("--artifact_type", default="model", help="Artifact type")
    log_p.add_argument("--project", required=True, help="W&B project")
    log_p.add_argument("--entity", default=None, help="W&B entity (user or org)")
    log_p.add_argument("--alias", default="latest", help="Artifact alias (e.g., latest)")
    log_p.add_argument("--netrc_path", default="/root/.netrc", help="Path to write .netrc")

    dl_p = subparsers.add_parser("load", help="Download a model artifact")
    dl_p.add_argument("--artifact_path", required=True, help="entity/project/name:version")
    dl_p.add_argument("--output_dir", default="./wandb_artifacts", help="Download directory")
    dl_p.add_argument("--filename", default=None, help="Optional file to print within artifact")
    dl_p.add_argument("--netrc_path", default="/root/.netrc", help="Path to write .netrc")

    args = parser.parse_args()

    if args.command == "log":
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(args.model_path)
        log_model_artifact(
            model_path=args.model_path,
            artifact_name=args.artifact_name,
            artifact_type=args.artifact_type,
            project=args.project,
            entity=args.entity,
            alias=args.alias,
            netrc_path=args.netrc_path,
        )
    elif args.command == "load":
        download_model_artifact(
            artifact_path=args.artifact_path,
            output_dir=args.output_dir,
            filename=args.filename,
            netrc_path=args.netrc_path,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
