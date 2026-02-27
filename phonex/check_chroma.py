import sqlite3
import json

CORRECT_CONFIG = json.dumps({
    "_type": "CollectionConfigurationInternal",
    "hnsw_configuration": {
        "_type": "HNSWConfigurationInternal",
        "space": "l2",
        "ef_construction": 100,
        "ef_search": 100,
        "num_threads": 4,
        "M": 16,
        "resize_factor": 1.2,
        "batch_size": 100,
        "sync_threshold": 1000
    }
})

conn = sqlite3.connect(r'D:\Gitrnd\phonex\indexfolder\phonex_index_download\chroma.sqlite3')
cur = conn.execute("SELECT id, name, config_json_str FROM collections")
rows = cur.fetchall()
print("Before:", rows)

for row in rows:
    conn.execute(
        "UPDATE collections SET config_json_str = ? WHERE id = ?",
        (CORRECT_CONFIG, row[0])
    )
conn.commit()

cur = conn.execute("SELECT id, name, config_json_str FROM collections")
print("After:", cur.fetchall())
conn.close()
print("Done!")
