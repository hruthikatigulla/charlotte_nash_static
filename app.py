import json
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recompute", methods=["GET"])
def recompute_static():
    facility = request.args.get("facility", "")

    # Nashville – BJJ
    if facility == "nash_bjj":
        with open("static/nash_bjj.json") as f:
            return jsonify(json.load(f))

    # Nashville – CrossFit
    if facility == "nash_cf":
        with open("static/nash_cf.json") as f:
            return jsonify(json.load(f))

    # Charlotte – Harris Teeter (NEW)
    if facility == "charlotte_ht":
        with open("static/charlotte_ht.json") as f:
            return jsonify(json.load(f))

    return jsonify({"ok": False, "error": f"Unknown facility '{facility}'"})


@app.get("/health")
def health():
    return jsonify({"ok": True, "message": "Static site selection demo running"})


# No dynamic blocks (optional)
@app.route("/blocks")
def blocks_disabled():
    return jsonify({"ok": False, "error": "Block groups disabled for demo"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
