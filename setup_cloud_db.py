"""
MongoDB Atlas (Cloud) Setup Script for DocBlaze
One command to setup everything!
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, TEXT
from urllib.parse import quote_plus
import os

# =====================================================
# CONFIGURATION - UPDATE THESE VALUES
# =====================================================

# Option 1: Direct connection string
MONGO_URL = os.getenv("MONGO_URL", "mongodb+srv://hritickra99_db_user:wXx0JRLwX1Kq8GP4@cluster0.r1fszik.mongodb.net/?appName=Cluster0")

# Option 2: Build from components (if you prefer)
# MONGO_USERNAME = "your_username"
# MONGO_PASSWORD = "your_password"
# MONGO_CLUSTER = "cluster0.xxxxx.mongodb.net"
# MONGO_URL = f"mongodb+srv://{quote_plus(MONGO_USERNAME)}:{quote_plus(MONGO_PASSWORD)}@{MONGO_CLUSTER}/?retryWrites=true&w=majority"

DB_NAME = "docblaze"

# =====================================================

async def setup_database():
    """Complete database setup for MongoDB Atlas"""
    
    print("🌐 DocBlaze - MongoDB Atlas Setup")
    print("=" * 60)
    
    # Connect to Atlas
    print(f"\n🔗 Connecting to MongoDB Atlas...")
    try:
        client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=10000)
        # Test connection
        await client.server_info()
        print("✅ Connected successfully!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPlease check:")
        print("1. Your connection string is correct")
        print("2. Your IP is whitelisted in Atlas (Network Access)")
        print("3. Your username/password are correct")
        return
    
    db = client[DB_NAME]
    print(f"📦 Using database: {DB_NAME}\n")
    
    # =====================================================
    # 1. USERS COLLECTION
    # =====================================================
    print("👥 Creating 'users' collection...")
    
    if "users" not in await db.list_collection_names():
        await db.create_collection("users")
    
    users = db.users
    
    # Drop existing indexes to avoid conflicts
    try:
        await users.drop_indexes()
    except:
        pass
    
    # Create indexes
    await users.create_index([("email", ASCENDING)], unique=True)
    await users.create_index([("user_id", ASCENDING)], unique=True)
    await users.create_index([("created_at", DESCENDING)])
    
    print("  ✅ Indexes created: email, user_id, created_at")
    
    # Schema validation
    try:
        await db.command({
            "collMod": "users",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["user_id", "username", "email", "password", "created_at"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "username": {"bsonType": "string", "minLength": 3, "maxLength": 50},
                        "email": {"bsonType": "string"},
                        "password": {"bsonType": "string"},
                        "created_at": {"bsonType": "date"}
                    }
                }
            }
        })
        print("  ✅ Schema validation applied\n")
    except Exception as e:
        print(f"  ⚠️  Schema validation skipped: {e}\n")
    
    # =====================================================
    # 2. SESSIONS COLLECTION
    # =====================================================
    print("🔐 Creating 'sessions' collection...")
    
    if "sessions" not in await db.list_collection_names():
        await db.create_collection("sessions")
    
    sessions = db.sessions
    
    try:
        await sessions.drop_indexes()
    except:
        pass
    
    await sessions.create_index([("session_id", ASCENDING)], unique=True)
    await sessions.create_index([("user_id", ASCENDING)])
    await sessions.create_index([("last_active", DESCENDING)])
    await sessions.create_index([("user_id", ASCENDING), ("last_active", DESCENDING)])
    
    print("  ✅ Indexes created: session_id, user_id, last_active")
    
    try:
        await db.command({
            "collMod": "sessions",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["session_id", "user_id", "session_name", "created_at", "last_active"],
                    "properties": {
                        "session_id": {"bsonType": "string"},
                        "user_id": {"bsonType": "string"},
                        "session_name": {"bsonType": "string", "minLength": 1, "maxLength": 100},
                        "created_at": {"bsonType": "date"},
                        "last_active": {"bsonType": "date"}
                    }
                }
            }
        })
        print("  ✅ Schema validation applied\n")
    except Exception as e:
        print(f"  ⚠️  Schema validation skipped: {e}\n")
    
    # =====================================================
    # 3. DOCUMENTS COLLECTION
    # =====================================================
    print("📄 Creating 'documents' collection...")
    
    if "documents" not in await db.list_collection_names():
        await db.create_collection("documents")
    
    documents = db.documents
    
    try:
        await documents.drop_indexes()
    except:
        pass
    
    await documents.create_index([("doc_id", ASCENDING)], unique=True)
    await documents.create_index([("user_id", ASCENDING)])
    await documents.create_index([("session_id", ASCENDING)])
    await documents.create_index([("session_id", ASCENDING), ("uploaded_at", DESCENDING)])
    await documents.create_index([("boq_data.is_boq", ASCENDING)])
    await documents.create_index([("has_tables", ASCENDING)])
    await documents.create_index([("extracted_text", TEXT)])
    
    print("  ✅ Indexes created: doc_id, session_id, boq_data.is_boq, text search")
    
    try:
        await db.command({
            "collMod": "documents",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["doc_id", "user_id", "session_id", "filename", "file_path", "file_type", "uploaded_at"],
                    "properties": {
                        "doc_id": {"bsonType": "string"},
                        "user_id": {"bsonType": "string"},
                        "session_id": {"bsonType": "string"},
                        "filename": {"bsonType": "string"},
                        "file_path": {"bsonType": "string"},
                        "file_type": {
                            "bsonType": "string",
                            "enum": ["pdf", "docx", "xlsx", "xls", "csv", "jpg", "jpeg", "png", "bmp", "tiff"]
                        },
                        "uploaded_at": {"bsonType": "date"}
                    }
                }
            }
        })
        print("  ✅ Schema validation applied\n")
    except Exception as e:
        print(f"  ⚠️  Schema validation skipped: {e}\n")
    
    # =====================================================
    # 4. CHAT_HISTORY COLLECTION
    # =====================================================
    print("💬 Creating 'chat_history' collection...")
    
    if "chat_history" not in await db.list_collection_names():
        await db.create_collection("chat_history")
    
    chat_history = db.chat_history
    
    try:
        await chat_history.drop_indexes()
    except:
        pass
    
    await chat_history.create_index([("chat_id", ASCENDING)], unique=True)
    await chat_history.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])
    await chat_history.create_index([("user_id", ASCENDING)])
    await chat_history.create_index([("timestamp", DESCENDING)])
    await chat_history.create_index([("query", TEXT), ("response", TEXT)])
    
    print("  ✅ Indexes created: chat_id, session_id+timestamp, text search")
    
    try:
        await db.command({
            "collMod": "chat_history",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["chat_id", "session_id", "user_id", "query", "response", "timestamp"],
                    "properties": {
                        "chat_id": {"bsonType": "string"},
                        "session_id": {"bsonType": "string"},
                        "user_id": {"bsonType": "string"},
                        "query": {"bsonType": "string"},
                        "response": {"bsonType": "string"},
                        "timestamp": {"bsonType": "date"}
                    }
                }
            }
        })
        print("  ✅ Schema validation applied\n")
    except Exception as e:
        print(f"  ⚠️  Schema validation skipped: {e}\n")
    
    # =====================================================
    # 5. BOQ_DATA COLLECTION
    # =====================================================
    print("📊 Creating 'boq_data' collection...")
    
    if "boq_data" not in await db.list_collection_names():
        await db.create_collection("boq_data")
    
    boq_data = db.boq_data
    
    try:
        await boq_data.drop_indexes()
    except:
        pass
    
    await boq_data.create_index([("doc_id", ASCENDING)])
    await boq_data.create_index([("session_id", ASCENDING)])
    await boq_data.create_index([("user_id", ASCENDING)])
    await boq_data.create_index([("created_at", DESCENDING)])
    await boq_data.create_index([("boq_items.item_description", TEXT)])
    
    print("  ✅ Indexes created: doc_id, session_id, text search on items")
    
    try:
        await db.command({
            "collMod": "boq_data",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["doc_id", "session_id", "user_id", "boq_items", "created_at"],
                    "properties": {
                        "doc_id": {"bsonType": "string"},
                        "session_id": {"bsonType": "string"},
                        "user_id": {"bsonType": "string"},
                        "boq_items": {"bsonType": "array"},
                        "created_at": {"bsonType": "date"}
                    }
                }
            }
        })
        print("  ✅ Schema validation applied\n")
    except Exception as e:
        print(f"  ⚠️  Schema validation skipped: {e}\n")
    
    # =====================================================
    # SUMMARY
    # =====================================================
    print("=" * 60)
    print("✨ MongoDB Atlas setup complete!")
    print("=" * 60)
    
    collections = await db.list_collection_names()
    print(f"\n📊 Collections created ({len(collections)}):")
    for coll_name in collections:
        coll = db[coll_name]
        count = await coll.count_documents({})
        indexes = await coll.index_information()
        print(f"  • {coll_name}: {count} documents, {len(indexes)} indexes")
    
    print(f"\n🎯 Database ready: {DB_NAME}")
    print(f"🌐 Connection: {MONGO_URL.split('@')[1].split('/')[0]}")  # Hide credentials
    print("\n✅ You can now run your DocBlaze backend with:")
    print(f'   export MONGO_URL="{MONGO_URL}"')
    print("   python backend.py")
    
    client.close()

def main():
    """Run setup"""
    print("\n" + "="*60)
    print("  DocBlaze - MongoDB Atlas Setup Script")
    print("="*60 + "\n")
    
    print("⚠️  Before running, ensure:")
    print("1. You've updated MONGO_URL in this script")
    print("2. Your IP is whitelisted in Atlas (Network Access)")
    print("3. Your database user has read/write permissions\n")
    
    response = input("Ready to setup database? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        asyncio.run(setup_database())
    else:
        print("\n❌ Setup cancelled")

if __name__ == "__main__":
    main()