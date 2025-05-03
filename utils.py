def format_score(score):
    if score > 75:
        return f"✅ {score} (Excellent)"
    elif score > 50:
        return f"⚠️ {score} (Good)"
    else:
        return f"❌ {score} (Low)"
