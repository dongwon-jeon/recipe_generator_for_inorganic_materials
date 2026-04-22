#!/usr/bin/env python3 
from dotenv import load_dotenv

# load_dotenv()
HARDCODED_OPENAI_KEY = "Enter your own API KEY"


import streamlit as st
import os
import re
from experiment.predict import RAGRecipePredictor, RecipePredictor
from litellm import embedding
import litellm
from pdf2recipe import pdf_bytelist_to_recipes
from litellm import completion
import openai

# Automatically drop unsupported params for GPT-5 models
litellm.drop_params = True

st.set_page_config(
    page_title="Recipe Recommendations for Inorganic Solid-State Synthesis",
    page_icon=":microscope:",
    layout="wide",
)

st.title("Recipe Recommendations for Inorganic Solid-State Synthesis")
# Input fields
st.sidebar.title("Input Parameters")

# Select API Provider
api_provider = st.session_state.get("api_provider", "OpenAI")
api_provider = st.sidebar.radio(
    "API Provider",
    ["OpenAI", "OpenRouter"],
    index=0 if api_provider == "OpenAI" else 1,
    key="api_provider_selector"
)

# Set default key (Hardcoded)
st.session_state.api_provider = st.session_state.get("api_provider_selector", "OpenAI")

def sanitize_api_key(api_key: str) -> str:
    if api_key is None:
        return ""
    return re.sub(r"\s+", "", api_key).strip()

def is_invalid_api_key(api_key: str) -> bool:
    clean_api_key = sanitize_api_key(api_key)

    invalid_values = {
        "",
        "empty",
        "EnteryourownAPIKEY",
        "PleaseinputyourOpenAIAPIkey.",
        "PleaseinputyourOpenRouterAPIkey.",
    }

    return clean_api_key in invalid_values

# Set default key by provider
if st.session_state.api_provider == "OpenAI":
    # If the key is not in session_state, initialize with the hardcoded key
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = HARDCODED_OPENAI_KEY

    default_key = st.session_state.openai_key
    litellm.openai_key = st.session_state.openai_key
    litellm.api_key = st.session_state.openai_key

    if not default_key or default_key == "empty":
        default_key = "Please input your OpenAI API key."
    api_key_input = st.sidebar.text_input("OpenAI API Key", default_key, type="password")
else:  # OpenRouter
    default_key = st.session_state.get("openrouter_key", os.environ.get("OPENROUTER_API_KEY", ""))
    if not default_key or default_key == "empty":
        default_key = "Please input your OpenRouter API key."
    api_key_input = st.sidebar.text_input("OpenRouter API Key", default_key, type="password")

update_key = st.sidebar.button("Update")

if update_key:
    clean_api_key = sanitize_api_key(api_key_input)

    if is_invalid_api_key(clean_api_key):
        st.sidebar.error("Please enter a valid API key.")
        st.stop()

    if api_provider == "OpenAI":
        st.session_state.openai_key = clean_api_key
        st.session_state.api_provider = "OpenAI"
        litellm.openai_key = clean_api_key
        litellm.api_key = clean_api_key
        if hasattr(litellm, "openrouter_key"):
            litellm.openrouter_key = None
    else:
        st.session_state.openrouter_key = clean_api_key
        st.session_state.api_provider = "OpenRouter"
        litellm.openrouter_key = clean_api_key
        litellm.api_key = clean_api_key

    if clean_api_key != api_key_input:
        st.session_state.api_key_info_message = "Whitespace characters in the API key were removed automatically."

    st.session_state.api_key_success_message = f"{api_provider} API key was updated successfully."

    try:
        get_predictors.clear()
    except NameError:
        pass

    if hasattr(st.session_state, "messages"):
        del st.session_state.messages
    if hasattr(st.session_state, "references"):
        del st.session_state.references
    if hasattr(st.session_state, "retrieval_info"):
        del st.session_state.retrieval_info

    st.rerun()

# Set the currently used provider and key
current_provider = st.session_state.get("api_provider", "OpenAI")
if current_provider == "OpenAI":
    current_api_key = st.session_state.get("openai_key", os.environ.get("OPENAI_API_KEY", ""))
else:
    current_api_key = st.session_state.get("openrouter_key", os.environ.get("OPENROUTER_API_KEY", ""))

# Show a warning if there is no API key
if not current_api_key or current_api_key in ["Please input your OpenAI API key.", "Please input your OpenRouter API key.", "empty"]:
    st.sidebar.warning(f"Please set your {current_provider} API key above.")

if "api_key_success_message" in st.session_state:
    st.sidebar.success(st.session_state.api_key_success_message)
    del st.session_state.api_key_success_message

if "api_key_info_message" in st.session_state:
    st.sidebar.info(st.session_state.api_key_info_message)
    del st.session_state.api_key_info_message

# Model selection (applies immediately outside the form)
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Settings")

# Switch model options by provider
if current_provider == "OpenRouter":
    model_options = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "openai/gpt-4-turbo",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        "anthropic/claude-haiku-4.5",
        "google/gemini-pro",
        "meta-llama/llama-3.1-70b-instruct",
    ]
    model = st.sidebar.selectbox("Model", model_options, key="model_select")
    # Allow custom model input
    custom_model = st.sidebar.text_input("Or enter custom model (e.g., anthropic/claude-haiku-4.5)", "")
    if custom_model:
        model = custom_model
else:  # OpenAI
    model_options = [
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-5",
        "o3",
        "o3-mini",
        "o3-mini-low",
        "o3-mini-high"
    ]
    model = st.sidebar.selectbox("Model", model_options, index=0, key="model_select")  # gpt-4.1-mini is the default

    # Additional parameters for GPT-5 family (reasoning model)
    if model.startswith("gpt-5"):
        st.sidebar.markdown("#### GPT-5 Parameters (Reasoning Model)")
        # st.sidebar.info("GPT-5 is a reasoning model and does not support temperature, top_p, etc.")

        reasoning_effort = st.sidebar.select_slider(
            "Reasoning Effort",
            options=["low", "medium", "high"],
            value=st.session_state.get("gpt5_reasoning_effort", "medium"),
            help="Controls how deeply the model thinks about the problem",
            key="reasoning_effort_slider"
        )
        text_verbosity = st.sidebar.select_slider(
            "Text Verbosity",
            options=["low", "medium", "high"],
            value=st.session_state.get("gpt5_text_verbosity", "medium"),
            help="Controls how detailed the output should be",
            key="text_verbosity_slider"
        )
        # Save to session_state
        st.session_state.gpt5_reasoning_effort = reasoning_effort
        st.session_state.gpt5_text_verbosity = text_verbosity
    else:
        # Default values when not using GPT-5
        if "gpt5_reasoning_effort" not in st.session_state:
            st.session_state.gpt5_reasoning_effort = "medium"
        if "gpt5_text_verbosity" not in st.session_state:
            st.session_state.gpt5_text_verbosity = "medium"

st.sidebar.markdown("---")

# Remaining input fields inside a form
with st.sidebar, st.form("recipe_form"):
    material_name = st.text_input("Material Name", "LiGa(SeO3)2")
    synthesis_technique = st.text_input("Synthesis Technique", "mechanochemical process")
    application = st.text_input("Application", "solid-state electrolyte")
    other_contstraints = st.text_area("Other Constraints", "")

    top_k = st.slider("Number of Retrievals", 0, 10, 5)

    files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    if files:
        for file in files:
            st.write(file.name)

    generate_btn = st.form_submit_button("Recommend")


if generate_btn and is_invalid_api_key(current_api_key):
    st.warning("Please enter your API key and click 'Update' before generating a recipe.")
    st.stop()

clear_btn = st.sidebar.button("Clear Conversation")
if clear_btn:
    if hasattr(st.session_state, "messages"):
        del st.session_state.messages
    if hasattr(st.session_state, "references"):
        del st.session_state.references
    if hasattr(st.session_state, "retrieval_info"):
        del st.session_state.retrieval_info
    st.rerun()

if not generate_btn and not hasattr(st.session_state, "messages"):
#    st.write("This is a demo of the Materials Synthesis Recipe Recommender. Please enter the desired material properties and click on the 'Recommend' button to get a list of materials synthesis recipes that can be used to synthesize materials with the desired properties.")
    st.write(
                    """
This page generates literature-guided synthesis recipes for inorganic solid-state materials using a retrieval-augmented generation pipeline.

For detailed documentation, source code, and implementation updates, see the GitHub repository:
https://github.com/dongwon-jeon/ssr_recipe_generator_for_inorganic_materials

1. Enter your personal `OpenAI` API key.
   Make sure your API account has access to the model you want to use.

2. Check that the models supported in this demo are enabled for your OpenAI project.
   In the OpenAI Platform, go to `Settings > {your project} > Limits` and confirm that the models used in this demo are available under your project settings.

   ```python
   available_model_list = [
       "gpt-4.1-mini",
       "gpt-4o-mini",
       "gpt-5.2",
       "gpt-5-mini",
       "gpt-5",
       "o3",
       "o3-mini",
       "o3-mini-low",
       "o3-mini-high"
   ]

3. Click the `Update` button after entering the API key.

4. Fill in the prediction inputs.
   - `Material Name`: target compound or composition
   - `Synthesis Technique`: process type to guide recipe generation
   - `Application`: intended use of the material, used to retrieve similar literature examples

5. Optionally adjust `Number of Retrievals`.
   - This controls how many retrieved literature recipes are used as reference exemplars for generation.

6. Optionally upload additional reference papers in PDF format.
   - Uploaded PDFs are parsed and used as additional user-provided references during recipe generation.

7. Click the `Recommend` button to generate a recipe.

8. The generated output includes:
   - target materials
   - precursor list
   - stepwise synthesis recipe

9. A `retrieval confidence` score is displayed with the generated recipe.
   - This score indicates how well the retrieved reference recipes match the input query.

10. Use the chat box below the generated response for follow-up questions or recipe revision requests.

11. The generated recipe is a literature-guided suggestion and should be experimentally validated.
"""
    )
    st.stop()

use_rag = top_k >= 1
output_filename = f"data/recipes.jsonl"

PREDICTION_PROMPT = """## Key Contributions
- **Novel materials or compounds**: {material_name}
- **Unique synthesis methods**: {synthesis_technique}
- **Specific applications or domains**: {application}
""".strip()

def get_embedding(contributions, provider="OpenAI", api_key=None):
    try:
        # Ensure UTF-8 encoding/decoding to avoid character-encoding issues
        if isinstance(contributions, str):
            contributions = contributions.encode('utf-8', errors='ignore').decode('utf-8')

        # For OpenRouter, call the OpenRouter API via the OpenAI SDK directly
        if provider == "OpenRouter":
            # Call OpenRouter API directly
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            response = client.embeddings.create(
                model="openai/text-embedding-3-large",
                input=[contributions],
                encoding_format="float"
            )
            emb = response.data[0].embedding
        else:
            # For OpenAI, use litellm
            embedding_model = 'text-embedding-3-large'
            if api_key:
                response = embedding(
                    model=embedding_model,
                    input=[contributions],
                    api_key=api_key,
                    encoding_format="float"
                )
            else:
                response = embedding(
                    model=embedding_model,
                    input=[contributions],
                    encoding_format="float"
                )
            emb = response['data'][0]['embedding']
        return emb
    except Exception as e:
        error_msg = str(e)
        # Detailed logging for debugging
        print(f"[Embedding Error] Provider: {provider}")
        print(f"[Embedding Error] Contributions preview: {contributions[:100] if contributions else 'None'}")
        print(f"[Embedding Error] Error type: {type(e).__name__}")
        print(f"[Embedding Error] Full error: {e}")

        if "invalid_api_key" in error_msg or "401" in error_msg:
            raise Exception(f"{provider} API key is invalid. Please enter a valid API key.")
        else:
            raise Exception(f"Error occurred while generating embeddings: {error_msg}")

def format_model_name(model_name, provider):
    """Format the model name appropriately based on the provider."""
    if provider == "OpenRouter":
        # OpenRouter format: openrouter/provider/model-name
        # e.g., openrouter/anthropic/claude-haiku-4.5, openrouter/openai/gpt-5-pro
        # If already prefixed with openrouter/, keep as-is
        if model_name.startswith("openrouter/"):
            return model_name
        # If provider/model format (e.g., anthropic/claude-haiku-4.5), prepend openrouter/
        elif "/" in model_name and not model_name.startswith("openrouter/"):
            return f"openrouter/{model_name}"
        # If only a bare model name (e.g., gpt-4o-mini), prepend openrouter/openai/ (default)
        else:
            return f"openrouter/openai/{model_name}"
    return model_name

@st.cache_resource
def get_predictors(_provider, _api_key, _model, _top_k, _gpt5_reasoning_effort="medium", _gpt5_text_verbosity="medium"):
    """Cache key includes provider and API key."""
    formatted_model = format_model_name(_model, _provider)
    rag_predictor = RAGRecipePredictor(
        model=formatted_model,
        prompt_filename="experiment/prompts/rag.txt",
        rag_topk=_top_k,
        retrieval_split="all",
        api_key=_api_key,
        gpt5_reasoning_effort=_gpt5_reasoning_effort,
        gpt5_text_verbosity=_gpt5_text_verbosity
    )
    base_predictor = RecipePredictor(
        model=formatted_model,
        prompt_filename="experiment/prompts/prediction.txt",
        api_key=_api_key,
        gpt5_reasoning_effort=_gpt5_reasoning_effort,
        gpt5_text_verbosity=_gpt5_text_verbosity
    )
    return rag_predictor, base_predictor

# Invalidate cache when the provider changes
previous_provider = st.session_state.get("previous_provider", current_provider)
if previous_provider != current_provider:
    get_predictors.clear()
    st.session_state.previous_provider = current_provider

# Initialize predictors
formatted_model = format_model_name(model, current_provider)
# Get GPT-5 parameters
gpt5_reasoning_effort = st.session_state.get("gpt5_reasoning_effort", "medium")
gpt5_text_verbosity = st.session_state.get("gpt5_text_verbosity", "medium")
rag_predictor, base_predictor = get_predictors(current_provider, current_api_key, model, top_k, gpt5_reasoning_effort, gpt5_text_verbosity)

def predict_recipe(material_name, synthesis_technique, application, other_contstraints, top_k, model, use_rag, provider, files=None, api_key=None):
    contributions = PREDICTION_PROMPT.format(
        material_name=material_name,
        synthesis_technique=synthesis_technique,
        application=application,
    )

    if use_rag or files:
        predictor = rag_predictor
        emb = get_embedding(contributions, provider, api_key=api_key)
        predictor.last_retrieval_info = None
        predictor.last_prompt = None
    else:
        predictor = base_predictor
        emb = None
        predictor.last_prompt = None

    if files:
        with st.spinner("Extracting recipes from PDFs..."):
            references = pdf_bytelist_to_recipes([file.read() for file in files])
    else:
        references = None

    predictor.base_references = references
    predictor.model = format_model_name(model, provider)

    if other_contstraints:
        contributions += f"\n\n## Other Constraints\n{other_contstraints}"

    batch = [
        {
            "contribution": contributions,
            "recipe": "",
            "contributions_embedding": emb
        }
    ]

    output = None
    try:
        import sys
        print(f"[Recipe Generation] Starting prediction with model: {predictor.model}", file=sys.stderr)
        print(f"[Recipe Generation] Provider: {provider}", file=sys.stderr)
        if predictor.model.startswith("gpt-5"):
            print(f"[Recipe Generation] GPT-5 reasoning effort: {predictor.gpt5_reasoning_effort}", file=sys.stderr)
            print(f"[Recipe Generation] GPT-5 text verbosity: {predictor.gpt5_text_verbosity}", file=sys.stderr)
        print(f"[Recipe Generation] API key (last 4 chars): ...{api_key[-4:] if api_key else 'None'}", file=sys.stderr)

        for _, output in predictor.predict(batch):
            print(f"[Recipe Generation] Generated output length: {len(output) if output else 0}", file=sys.stderr)
            if output:
                print(f"[Recipe Generation] Output preview: {output[:100]}...", file=sys.stderr)

        if not output:
            print("[Recipe Generation] WARNING: Output is empty or None!", file=sys.stderr)
            output = "Recipe generation failed. Please check your API key and model settings."
    except Exception as e:
        import traceback
        import sys
        error_details = traceback.format_exc()
        print(f"[Recipe Generation] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"[Recipe Generation] Full traceback:", file=sys.stderr)
        print(error_details, file=sys.stderr)
        output = f"Error occurred during recipe generation: {type(e).__name__}: {str(e)}\n\nDetailed error:\n{error_details}"

    user_prompt = predictor.last_prompt
    if user_prompt is None:
        user_prompt = predictor.build_prompt(batch[0])[0]["content"]

    retrieval_info = None

    if use_rag or files:
        ref_outputs = []

        if references:
            ref_outputs.extend(references)

        raw_retrieval_info = getattr(predictor, "last_retrieval_info", None)
        retrieval_rows = raw_retrieval_info["rows"] if raw_retrieval_info else None

        if retrieval_rows:
            num_refs = min(top_k, len(retrieval_rows.get("id", [])))
            for i in range(num_refs):
                rid = retrieval_rows["id"][i]
                contribution_text = retrieval_rows["contribution"][i]
                recipe_text = retrieval_rows["recipe"][i]

                ref_output = f"Semantic Scholar: [{rid}](https://www.semanticscholar.org/paper/{rid})\n"
                ref_output += f"{contribution_text}\n\n{recipe_text}"
                ref_outputs.append(ref_output)

            retrieval_info = {
                "retrieval_confidence": raw_retrieval_info.get("retrieval_confidence"),
                "retrieval_query_proximity": raw_retrieval_info.get("retrieval_query_proximity"),
                "retrieval_exemplar_consistency": raw_retrieval_info.get("retrieval_exemplar_consistency"),
                "retrieval_top1_similarity": raw_retrieval_info.get("retrieval_top1_similarity"),
            }

        references = ref_outputs
    else:
        references = None

    return output, references, user_prompt, retrieval_info

if not hasattr(st.session_state, "messages"):
    st.session_state.messages = []

    with st.spinner("Generating recipes..."):
        try:
            recipe, references, user_prompt, retrieval_info = predict_recipe(
                material_name,
                synthesis_technique,
                application,
                other_contstraints,
                top_k,
                model,
                use_rag,
                current_provider,
                files=files,
                api_key=current_api_key
            )
            st.session_state.references = references
            st.session_state.retrieval_info = retrieval_info
            st.session_state.messages.append({
                "role": "user",
                "content": user_prompt,
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": recipe
            })
        except Exception as e:
            error_msg = str(e)
            if "invalid_api_key" in error_msg or "401" in error_msg:
                st.error(f"API key error: {current_provider} API key is invalid. Please enter a valid API key in the left sidebar.")
            else:
                st.error(f"Error occurred: {error_msg}")
            st.warning("Please resolve the issue and try again.")
            st.stop()
else:
    if len(st.session_state.messages) >= 2 and hasattr(st.session_state, "references"):
        recipe = st.session_state.messages[1]["content"]
        references = st.session_state.references
        retrieval_info = st.session_state.get("retrieval_info", None)
        user_prompt = st.session_state.messages[0]["content"]
    else:
        st.session_state.messages = []
        st.rerun()

with st.chat_message("assistant"):
    st.header("Predicted Recipes")

    if retrieval_info and retrieval_info.get("retrieval_confidence") is not None:
        st.markdown(f"**Retrieval confidence:** {retrieval_info['retrieval_confidence']:.3f}")

        with st.expander("How is retrieval confidence calculated?", expanded=False):
            st.markdown("The retrieval confidence quantifies how well the retrieved reference recipes support the input query. It combines two factors. First, query-to-exemplar proximity measures whether the retrieved exemplars are semantically close to the input contribution. Second, exemplar-to-exemplar consistency measures whether the retrieved exemplars form a coherent reference group rather than a mixture of loosely related cases. Both terms are computed from cosine similarities between normalized contribution embeddings and are equally weighted in the final score. A higher retrieval confidence therefore indicates that the model retrieved references that are both relevant to the query and mutually consistent with one another. ")
            st.latex(r"\mathrm{sim}_i = \hat{q} \cdot \hat{e}_i")
            st.latex(r"R_{\mathrm{query}} = \sum_i w_i\,\mathrm{sim}_i")
            st.latex(r"R_{\mathrm{cluster}} = \frac{1}{\binom{k}{2}} \sum_{i<j} \hat{e}_i \cdot \hat{e}_j")
            st.latex(r"Q = \frac{R_{\mathrm{query}} + 1}{2}, \quad C = \frac{R_{\mathrm{cluster}} + 1}{2}")
            st.latex(r"R = 0.5Q + 0.5C")

    st.markdown(recipe)

    st.write("\n\n")

    if use_rag:
        st.header("References")
        for i, ref in enumerate(references):
            with st.expander(f"Reference {i + 1}", expanded=False):
                st.markdown(ref)

if len(st.session_state.messages) > 2:
    for message in st.session_state.messages[2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Ask a question about the recipe")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.spinner("Generating response..."):
        try:
            # Format the model name for the provider
            chat_model = format_model_name(model, current_provider)

            # GPT-5 family handling (reasoning model)
            if model.startswith("gpt-5"):
                # Get parameters from session_state
                reasoning_effort = st.session_state.get("gpt5_reasoning_effort", "medium")
                text_verbosity = st.session_state.get("gpt5_text_verbosity", "medium")

                response = completion(
                    model=chat_model,
                    messages=st.session_state.messages,
                    max_completion_tokens=16384,
                    reasoning_effort=reasoning_effort,
                    verbosity=text_verbosity
                )

            # o1/o3 family handling (reasoning models)
            elif model.startswith("o1") or model.startswith("o3"):
                completion_kwargs = {
                    "model": chat_model,
                    "messages": st.session_state.messages,
                    "max_completion_tokens": 16384
                }

                # Set reasoning_effort
                if model == "o3-mini-high":
                    completion_kwargs["reasoning_effort"] = "high"
                    completion_kwargs["model"] = chat_model.replace("-high", "")
                elif model == "o3-mini-low":
                    completion_kwargs["reasoning_effort"] = "low"
                    completion_kwargs["model"] = chat_model.replace("-low", "")
                elif model == "o3-mini":
                    completion_kwargs["reasoning_effort"] = "medium"

                response = completion(**completion_kwargs)

            # Standard GPT models (gpt-4, etc.)
            else:
                response = completion(
                    model=chat_model,
                    messages=st.session_state.messages,
                    max_tokens=4096,
                    temperature=0.0
                )

            with st.chat_message("assistant"):
                st.markdown(response["choices"][0]["message"]["content"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["choices"][0]["message"]["content"]
            })
        except Exception as e:
            error_msg = str(e)
            with st.chat_message("assistant"):
                if "invalid_api_key" in error_msg or "401" in error_msg:
                    st.error(f"API key error: {current_provider} API key is invalid.")
                else:
                    st.error(f"Error occurred: {error_msg}")
            # Remove the failed user message
            st.session_state.messages.pop()
