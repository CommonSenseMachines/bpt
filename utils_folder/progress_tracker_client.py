import requests


def reset():
    try:
        requests.post("http://localhost:9546/reset")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with progress tracker: {str(e)}")


def checkpoint(name: str, checkpoint_percent_done: float = None) -> dict:
    """
    Send a checkpoint update to the progress tracker server.

    Args:
        name: Name of the checkpoint
        checkpoint_percent_done: Optional percentage done within current checkpoint (0-1)

    Returns:
        dict with percent_done, estimated_time_until_next_checkpoint, and next_checkpoint_percent_done
    """
    try:
        response = requests.post(
            "http://localhost:9546/checkpoint",
            json={"name": name, "checkpoint_percent_done": checkpoint_percent_done},
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with progress tracker: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    print(checkpoint("0.25"))  # Hit first checkpoint
    print(checkpoint("0.5"))  # Hit second checkpoint
    print(checkpoint("0.75"))  # Hit third checkpoint
    print(checkpoint("1"))  # Hit final checkpoint
