import hashlib

def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()
