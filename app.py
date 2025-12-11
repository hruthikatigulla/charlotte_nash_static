import json
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------- HOME ----------
@app.route("/")
def home():
    return render_template("index.html")

# ---------- STATIC RESULTS ----------
@app.route("/recompute")
def recompute_static():
    facility = request.args.get("facility", "")

    file_map = {
        "nash_bjj": "static/nash_bjj.json",
        "nash_cf": "static/nash_cf.json",
        "charlotte_ht": "static/charlotte_ht.json",
    }

    if facility not in file_map:
        return jsonify({"ok": False, "error": "Unknown facility"})

    with open(file_map[facility]) as f:
        return jsonify(json.load(f))

# ---------- BLOCK GROUPS ----------
@app.route("/blocks")
def blocks_static():
    facility = request.args.get("facility", "")

    block_map = {
        "nash_bjj": "static/blocks_nashville.json",
        "nash_cf": "static/blocks_nashville.json",
        "charlotte_ht": "static/blocks_charlotte.json",
    }

    if facility not in block_map:
        return jsonify({"ok": False})

    with open(block_map[facility]) as f:
        return jsonify(json.load(f))

# ---------- HEALTH ----------
@app.route("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
