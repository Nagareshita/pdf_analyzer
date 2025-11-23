from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass
class VLMParamSet:
    # Core generation params
    preset: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    do_sample: bool | None = None
    max_new_tokens: int | None = None
    min_new_tokens: int | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    num_beams: int | None = None
    length_penalty: float | None = None
    diversity_penalty: float | None = None
    early_stopping: bool | None = None

    def to_clean_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# Common language policy: English-only responses
LANGUAGE_POLICY_SUFFIX = (
    " Please respond in English only. Use proper English grammar and spelling."
)

# Default mapping from detected type -> parameter set + prompt template
DEFAULT_TYPE_MAP: Dict[str, Dict[str, Any]] = {
    # Tables: emphasize accuracy, longer output allowed, beam search
    "table": {
        "params": VLMParamSet(
            preset="ocr", do_sample=False, num_beams=5,
            length_penalty=1.1, no_repeat_ngram_size=4,
            repetition_penalty=1.05, min_new_tokens=80, max_new_tokens=1600,
            temperature=0.2, top_p=0.6, top_k=10
        ),
        "prompt": (
            "The following image is a table. Describe its content in 1-2 concise sentences. "
            "Summarize the columns and main variables, including specific units or ranges if visible. "
            "Do not reproduce or modify the table itself."
        ) + LANGUAGE_POLICY_SUFFIX,
    },
    # Analysis charts
    "contour_plot": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.25, top_p=0.7, top_k=20, min_new_tokens=40, max_new_tokens=240, no_repeat_ngram_size=3, repetition_penalty=1.03),
        "prompt": "This is a contour/isoline plot. Explain the axes, value meanings, and contour levels or important regions in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "vector_field_plot": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.25, top_p=0.7, top_k=20, min_new_tokens=40, max_new_tokens=240, no_repeat_ngram_size=3, repetition_penalty=1.03),
        "prompt": "This image shows a vector field (arrows). Describe the trends in direction and magnitude, and any distinctive regions in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    # Technical diagrams: block/circuit/mechanical/flow/P&ID
    "technical_diagram": {
        "params": VLMParamSet(
            preset="qa", do_sample=False, num_beams=5,
            temperature=0.2, top_p=0.7, top_k=20,
            min_new_tokens=64, max_new_tokens=360,
            no_repeat_ngram_size=4, repetition_penalty=1.06,
        ),
        "prompt": (
            "The following image is a technical diagram (block diagram/circuit diagram/mechanical diagram/P&ID, etc.). "
            "Describe the main elements, their relationships, and signal/flow directions in 1-2 sentences. "
            "Only mention numbers or units if clearly visible; avoid speculation."
        ) + LANGUAGE_POLICY_SUFFIX
    },
    # Plots / charts: axes, variables, trends
    "plot_graph": {
        "params": VLMParamSet(
            preset="qa", do_sample=False, num_beams=4,
            min_new_tokens=64, max_new_tokens=800,
            no_repeat_ngram_size=3, repetition_penalty=1.03,
            temperature=0.3, top_p=0.7, top_k=20
        ),
        "prompt": (
            "The following image is a graph/chart. Describe the axes, variables, and trends in 1-2 sentences. "
            "Point out any noticeable peaks, increases/decreases, or correlations."
        ) + LANGUAGE_POLICY_SUFFIX
    },
    # Natural photos / illustrations
    "natural_image": {
        "params": VLMParamSet(
            preset="balanced", do_sample=True,
            temperature=0.7, top_p=0.85, top_k=30,
            min_new_tokens=32, max_new_tokens=256
        ),
        "prompt": (
            "Describe the content of the following image in 1-2 concise sentences as a caption. "
            "Include the main objects, scene characteristics, and any actions if present."
        ) + LANGUAGE_POLICY_SUFFIX
    },
    # Strict OCR transcription
    "text_ocr": {
        "params": VLMParamSet(
            preset="ocr", do_sample=False, num_beams=5,
            min_new_tokens=100, max_new_tokens=2000,
            no_repeat_ngram_size=4, repetition_penalty=1.05,
        ),
        "prompt": (
            "Strictly transcribe the text readable from the following image. "
            "Preserve line breaks and spacing as much as possible. Do not add explanations or notes. "
            "Output text only."
        ) + LANGUAGE_POLICY_SUFFIX
    },
    # Formulas to LaTeX
    "formula": {
        "params": VLMParamSet(
            preset="ocr", do_sample=False, num_beams=6,
            min_new_tokens=80, max_new_tokens=800,
            no_repeat_ngram_size=4, repetition_penalty=1.05,
        ),
        "prompt": (
            "Describe the mathematical formulas in the image using LaTeX notation. Enclose each formula in $ delimiters. "
            "If there are multiple formulas, list them on separate lines. No explanation text needed."
        ) + LANGUAGE_POLICY_SUFFIX
    },
    # Handwriting OCR
    "handwriting_note": {
        "params": VLMParamSet(preset="ocr", do_sample=False, num_beams=6, min_new_tokens=100, max_new_tokens=2000, no_repeat_ngram_size=4, repetition_penalty=1.05),
        "prompt": "Strictly transcribe the handwritten text as accurately as possible. Mark unreadable portions with [?]. No explanations needed." + LANGUAGE_POLICY_SUFFIX,
    },
    # Specific charts
    "bar_chart": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.25, top_p=0.7, top_k=20, min_new_tokens=48, max_new_tokens=320, no_repeat_ngram_size=3, repetition_penalty=1.03),
        "prompt": "Describe the bar chart's axes, categories, and which categories show large differences. Summarize in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "line_chart": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.25, top_p=0.7, top_k=20, min_new_tokens=48, max_new_tokens=320, no_repeat_ngram_size=3, repetition_penalty=1.03),
        "prompt": "Describe the line chart's axes, trends, peaks, and increases/decreases in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "pie_chart": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.25, top_p=0.7, top_k=20, min_new_tokens=40, max_new_tokens=200, no_repeat_ngram_size=3, repetition_penalty=1.03),
        "prompt": "Describe the main proportions in the pie chart in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "heatmap": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.3, top_p=0.75, top_k=30, min_new_tokens=48, max_new_tokens=256, no_repeat_ngram_size=3, repetition_penalty=1.03),
        "prompt": "Describe the heatmap's axes, high/low concentration regions, and prominent patterns concisely." + LANGUAGE_POLICY_SUFFIX,
    },
    "network_graph": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.3, top_p=0.75, top_k=30, min_new_tokens=48, max_new_tokens=256, no_repeat_ngram_size=3, repetition_penalty=1.03),
        "prompt": "Describe the network graph's main nodes, clusters, and strong connections concisely." + LANGUAGE_POLICY_SUFFIX,
    },
    "timeline": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.25, top_p=0.7, top_k=20, min_new_tokens=40, max_new_tokens=240),
        "prompt": "Summarize the sequence and key points of important events on the timeline in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "infographic": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.4, top_p=0.8, top_k=30, min_new_tokens=48, max_new_tokens=256),
        "prompt": "Summarize the infographic's main topic and key points concisely." + LANGUAGE_POLICY_SUFFIX,
    },
    # Subtypes of technical
    "flowchart": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=5, temperature=0.2, top_p=0.7, top_k=20, min_new_tokens=48, max_new_tokens=280, repetition_penalty=1.06),
        "prompt": "Describe the flowchart's start/end points and main branches/process flow in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "circuit_diagram": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=5, temperature=0.2, top_p=0.7, top_k=20, min_new_tokens=64, max_new_tokens=360, no_repeat_ngram_size=4, repetition_penalty=1.06),
        "prompt": "Describe the circuit diagram's main elements (power supply/resistors/coils/switches, etc.) and connection relationships concisely." + LANGUAGE_POLICY_SUFFIX,
    },
    "block_diagram": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=5, temperature=0.2, top_p=0.7, top_k=20, min_new_tokens=64, max_new_tokens=360, repetition_penalty=1.06),
        "prompt": "Describe the block diagram's main components and signal input/output relationships in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "p_and_id_diagram": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=5, temperature=0.2, top_p=0.7, top_k=20, min_new_tokens=64, max_new_tokens=360, repetition_penalty=1.06),
        "prompt": "Describe the P&ID's main equipment and fluid flow/connection relationships concisely." + LANGUAGE_POLICY_SUFFIX,
    },
    "thermal_circuit": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=5, temperature=0.2, top_p=0.7, top_k=20, min_new_tokens=64, max_new_tokens=360, no_repeat_ngram_size=4, repetition_penalty=1.06),
        "prompt": "Describe the thermal circuit's elements (heat sources/thermal resistances/capacitances) and heat flow direction relationships in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "fluid_circuit": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=5, temperature=0.2, top_p=0.7, top_k=20, min_new_tokens=64, max_new_tokens=360, no_repeat_ngram_size=4, repetition_penalty=1.06),
        "prompt": "Describe the fluid circuit (piping)'s main equipment/valves and flow direction/connection relationships in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "floorplan_architecture": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.25, top_p=0.7, top_k=20, min_new_tokens=40, max_new_tokens=240),
        "prompt": "Describe the architectural drawing/floor plan. Explain the main rooms, wall/door layout, and distinctive sections concisely." + LANGUAGE_POLICY_SUFFIX,
    },
    # Screens / documents
    "code_snippet_screenshot": {
        "params": VLMParamSet(preset="code", do_sample=False, num_beams=4, min_new_tokens=64, max_new_tokens=512, no_repeat_ngram_size=4, repetition_penalty=1.07),
        "prompt": "Describe the purpose of the code in the image and language clues in 1-2 sentences (summary, not transcription)." + LANGUAGE_POLICY_SUFFIX,
    },
    "ui_screenshot": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, min_new_tokens=40, max_new_tokens=240),
        "prompt": "Describe the UI screenshot's screen type and main elements/state concisely." + LANGUAGE_POLICY_SUFFIX,
    },
    "webpage_screenshot": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, min_new_tokens=48, max_new_tokens=256),
        "prompt": "Describe the webpage's type (article/listing/top page, etc.) and main elements in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "scanned_document_page": {
        "params": VLMParamSet(preset="ocr", do_sample=False, num_beams=5, min_new_tokens=80, max_new_tokens=1200, no_repeat_ngram_size=4, repetition_penalty=1.05),
        "prompt": "Scanned document page. Briefly describe the structure such as chapter titles and columns (no transcription needed)." + LANGUAGE_POLICY_SUFFIX,
    },
    "form_document": {
        "params": VLMParamSet(preset="ocr", do_sample=False, num_beams=5, min_new_tokens=80, max_new_tokens=800),
        "prompt": "Describe the form/application type and input field characteristics in 1-2 sentences (do not generate personal information)." + LANGUAGE_POLICY_SUFFIX,
    },
    # Maps / geo
    "map_geographical": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, min_new_tokens=40, max_new_tokens=240),
        "prompt": "Map or geographical diagram. Describe the target region and highlighted elements concisely." + LANGUAGE_POLICY_SUFFIX,
    },
    # Chemistry
    "chemical_diagram": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=4, temperature=0.25, top_p=0.7, top_k=20, min_new_tokens=40, max_new_tokens=240, no_repeat_ngram_size=3, repetition_penalty=1.03),
        "prompt": "Chemical reaction formula/molecular structure diagram. Summarize the main molecules or reaction relationships in 1-2 sentences (avoid speculation)." + LANGUAGE_POLICY_SUFFIX,
    },
    # Modelica diagrams (annotation/diagram view)
    "modelica_diagram": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=5, temperature=0.2, top_p=0.7, top_k=20, min_new_tokens=64, max_new_tokens=360, no_repeat_ngram_size=4, repetition_penalty=1.06),
        "prompt": (
            "This is a Modelica diagram/annotation view. "
            "Describe the icon/component names (to the extent readable), connection line flows and directions, and main inputs/outputs in 1-2 sentences. "
            "Avoid speculating about internal behavior; concisely summarize only the visible elements."
        ) + LANGUAGE_POLICY_SUFFIX,
    },
    # Org / mind maps
    "org_mind_map": {
        "params": VLMParamSet(preset="qa", do_sample=False, num_beams=3, temperature=0.3, top_p=0.8, top_k=30, min_new_tokens=40, max_new_tokens=240),
        "prompt": "Organization chart/mind map. Describe the hierarchical relationships and grouping of main nodes in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    # Natural photo subtypes
    "landscape": {
        "params": VLMParamSet(preset="balanced", do_sample=True, temperature=0.6, top_p=0.85, top_k=30, min_new_tokens=32, max_new_tokens=200),
        "prompt": "Describe the main landscape elements (mountains/ocean/cityscape, etc.) and atmosphere of the photo in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
    "person_portrait": {
        "params": VLMParamSet(preset="balanced", do_sample=True, temperature=0.6, top_p=0.85, top_k=30, min_new_tokens=32, max_new_tokens=200),
        "prompt": "Describe characteristics of the person in the photo (age group/posture/expression/clothing, etc.) concisely without over-speculating." + LANGUAGE_POLICY_SUFFIX,
    },
    "animal_wildlife": {
        "params": VLMParamSet(preset="balanced", do_sample=True, temperature=0.6, top_p=0.85, top_k=30, min_new_tokens=32, max_new_tokens=200),
        "prompt": "Describe the animal species (if identifiable), posture, and environment in 1-2 sentences (be conservative with identification/speculation)." + LANGUAGE_POLICY_SUFFIX,
    },
    "object_product": {
        "params": VLMParamSet(preset="balanced", do_sample=True, temperature=0.6, top_p=0.85, top_k=30, min_new_tokens=32, max_new_tokens=200),
        "prompt": "Describe the object/product's appearance and usage clues in 1-2 sentences (avoid speculating on brand names)." + LANGUAGE_POLICY_SUFFIX,
    },
    "vehicle_transport": {
        "params": VLMParamSet(preset="balanced", do_sample=True, temperature=0.6, top_p=0.85, top_k=30, min_new_tokens=32, max_new_tokens=200),
        "prompt": "Describe the vehicle type (car/train/ship/aircraft, etc.) and situation in 1-2 sentences." + LANGUAGE_POLICY_SUFFIX,
    },
}


# Parameters for stage-1 classification (default)
# User-preferred fast/greedy setting
CLASSIFIER_PARAMS = VLMParamSet(
    preset="balanced",
    temperature=0.0,
    top_p=1.0,
    top_k=20,
    do_sample=False,
    max_new_tokens=96,
    min_new_tokens=16,
    repetition_penalty=1.02,
    no_repeat_ngram_size=3,
    num_beams=1,
    length_penalty=1.0,
    diversity_penalty=0.0,
    early_stopping=False,
)



def build_classifier_prompt(labels: List[str]) -> str:
    labels_txt = ", ".join(labels)
    guidance = (
        "Classification guidelines:\n"
        "- 'table' for tabular data with rows/columns\n"
        "- 'plot_graph' for graphs with axes and numerical values (generic graphs)\n"
        "- Specific chart types: 'bar_chart', 'line_chart', 'pie_chart', 'heatmap', 'contour_plot', 'vector_field_plot'\n"
        "- Technical diagrams: 'flowchart', 'circuit_diagram', 'block_diagram', 'p_and_id_diagram', 'modelica_diagram', 'technical_diagram' (generic)\n"
        "- Text content: 'text_ocr' (printed text), 'handwriting_note' (handwritten), 'formula' (mathematical formulas)\n"
        "- Documents: 'scanned_document_page', 'form_document', 'code_snippet_screenshot', 'ui_screenshot', 'webpage_screenshot'\n"
        "- Natural photos: 'natural_image' (generic), 'landscape', 'person_portrait', 'animal_wildlife', 'object_product', 'vehicle_transport'\n"
        "- Other: 'map_geographical', 'chemical_diagram', 'network_graph', 'timeline', 'infographic', 'org_mind_map', 'floorplan_architecture'\n"
        "Choose the MOST SPECIFIC type that matches. If uncertain, choose a more general category."
    )
    return (
        "Select and return the main type of the following image. "
        f"Candidates: [{labels_txt}].\n"
        f"{guidance}\n"
        "Output ONLY the following JSON: {\\\"type\\\": <one of the candidates>, \\\"confidence\\\": <0-1>, \\\"reason\\\": <short reason>}\n"
        "Please respond in English only."
    )


CLASSIFIER_PROMPT = build_classifier_prompt(list(DEFAULT_TYPE_MAP.keys()))
