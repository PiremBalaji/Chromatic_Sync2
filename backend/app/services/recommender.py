COLOR_PAIRS = {
    "black": ["white", "beige", "denim blue"],
    "white": ["black", "navy", "olive"],
    "gray": ["black", "burgundy", "white"],
    "red": ["black", "white", "dark denim"],
    "orange": ["cream", "brown", "dark denim"],
    "yellow": ["navy", "white", "gray"],
    "green": ["cream", "black", "brown"],
    "blue": ["white", "tan", "gray"],
    "purple": ["white", "charcoal", "light gray"],
    "pink": ["white", "gray", "navy"],
    "brown": ["cream", "black", "denim blue"],
    "beige": ["white", "brown", "black"],
}


def recommendations_for(category: str, color: str) -> list[dict[str, str]]:
    pair_colors = COLOR_PAIRS.get(color, ["black", "white", "denim blue"])
    if category == "top":
        return [
            {"type": "bottom", "item": "High-waisted trousers", "color": pair_colors[0], "reason": "Balances the silhouette"},
            {"type": "bottom", "item": "Straight-leg jeans", "color": pair_colors[1], "reason": "Easy everyday contrast"},
            {"type": "bottom", "item": "A-line skirt", "color": pair_colors[2], "reason": "Adds shape and movement"},
        ]
    if category == "bottom":
        return [
            {"type": "top", "item": "Fitted knit top", "color": pair_colors[0], "reason": "Keeps proportions clean"},
            {"type": "top", "item": "Relaxed cotton shirt", "color": pair_colors[1], "reason": "Creates a soft contrast"},
            {"type": "top", "item": "Layered lightweight blouse", "color": pair_colors[2], "reason": "Adds visual texture"},
        ]
    return [
        {"type": "styling", "item": "Minimal accessories", "color": pair_colors[0], "reason": "Lets the one-piece lead"},
        {"type": "styling", "item": "Neutral footwear", "color": pair_colors[1], "reason": "Grounds the complete look"},
        {"type": "styling", "item": "Contrasting outer layer", "color": pair_colors[2], "reason": "Adds a second visual layer"},
    ]
