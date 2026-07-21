import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS shoes (
    id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    category TEXT NOT NULL,
    gender TEXT NOT NULL,
    price NUMERIC NOT NULL,
    final_price NUMERIC NOT NULL,
    cost NUMERIC NOT NULL,
    gross_margin NUMERIC,
    gross_margin_pct NUMERIC,
    financial_tier TEXT NOT NULL,
    colors_available TEXT[] NOT NULL,
    performance_specs JSONB
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS inventory (
    id SERIAL PRIMARY KEY,
    shoe_id INTEGER REFERENCES shoes(id) ON DELETE CASCADE,
    color TEXT NOT NULL,
    size TEXT NOT NULL,
    stock INTEGER NOT NULL,
    image TEXT
);
""")
conn.commit()
print("Tables ready.")

# Safe to re-run: clears old data before reloading
cur.execute("DELETE FROM inventory;")
cur.execute("DELETE FROM shoes;")

with open("veloxa_enhanced_catalog.json", "r") as f:
    catalog = json.load(f)["catalog"]

for shoe in catalog:
    cur.execute(
        """INSERT INTO shoes
           (id, model, category, gender, price, final_price, cost,
            gross_margin, gross_margin_pct, financial_tier, colors_available, performance_specs)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            shoe["id"], shoe["model"], shoe["category"], shoe["gender"],
            shoe["price"], shoe["finalPrice"], shoe["cost"],
            shoe.get("gross_margin"), shoe.get("gross_margin_pct"),
            shoe["financial_tier"], shoe["colors_available"],
            json.dumps(shoe.get("performance_specs", {})),
        ),
    )
    for item in shoe["inventory"]:
        cur.execute(
            "INSERT INTO inventory (shoe_id, color, size, stock, image) VALUES (%s,%s,%s,%s,%s)",
            (shoe["id"], item["color"], item["size"], item["stock"], item["image"]),
        )

conn.commit()

cur.execute("SELECT COUNT(*) FROM shoes;")
shoe_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM inventory;")
inv_count = cur.fetchone()[0]
cur.execute("SELECT model, financial_tier, gender FROM shoes ORDER BY id LIMIT 3;")
samples = cur.fetchall()

print(f"\nMigrated: {shoe_count} shoes, {inv_count} inventory rows.")
print("Sample rows:")
for row in samples:
    print(f"  {row}")

cur.close()
conn.close()