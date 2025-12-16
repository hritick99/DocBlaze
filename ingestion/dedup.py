async def is_duplicate(documents_collection, user_id: str, file_hash: str) -> bool:
    existing = await documents_collection.find_one({
        "user_id": user_id,
        "file_hash": file_hash
    })
    return existing is not None
