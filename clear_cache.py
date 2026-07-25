import os
from dotenv import load_dotenv
from upstash_redis import Redis as UpstashRedis

load_dotenv()
redis_client = UpstashRedis.from_env()

keys = redis_client.keys("veloxa:cache:*")
for key in keys:
    redis_client.delete(key)
print(f"Cleared {len(keys)} cached entries.")