# PalgharGPT --- QLoRA Domain Adaptation for Palghar District

PalgharGPT is an experimental, domain-adapted language model built by
continuing causal language-model pretraining of
**Qwen2.5-1.5B-Instruct** on three Palghar district documents. The
project uses **4-bit NF4 quantization** and **QLoRA/LoRA adapters** to
adapt the base model to regional development-related language and
terminology.

> **Important:** This project uses continued causal-language-model
> pretraining. It is **not** supervised question-answer fine-tuning and
> it does **not** use Retrieval-Augmented Generation (RAG). The model
> does not retrieve source passages at answer time, and generated facts
> must be checked against the original documents.

## Project objectives

-   Adapt a pretrained language model to information and terminology
    related to Palghar district, Maharashtra.
-   Build a reproducible page-level text extraction and
    dataset-preparation workflow from PDF sources.
-   Fine-tune parameter-efficient LoRA adapters while keeping the base
    model quantized and frozen.
-   Record dataset provenance, page splits, training configuration, and
    evaluation metrics.

## Source documents

The notebook expects these three PDFs to be uploaded when prompted:

  -----------------------------------------------------------------------
  Document                            Publication / reference information
  ----------------------------------- -----------------------------------
  District Socio-Economic Review 2025 Publication year: 2025; primarily
                                      covers 2024--25, although
                                      individual tables may use different
                                      reference dates.

  Palghar District Strategic Plan     Publication year: 2024; contains
                                      multiple reference periods,
                                      including FY2016--FY2022.

  Palghar District Development        Research reference date stated in
  Research Report                     the report: 7 November 2026;
                                      includes indicators from different
                                      historical periods, report-level
                                      synthesis, and recommendations.
  -----------------------------------------------------------------------

The workflow preserves source filename, document title, publication
year, reference-period notes, and physical page number in the extracted
page records. Proposed projects in the research report should be treated
as recommendations, not as confirmed or approved government projects.

**Data note:** Ensure you have the right to use and redistribute the
source documents and any extracted text before publishing them. Do not
upload source PDFs or derived datasets publicly unless their terms
permit it.

## Technical approach

1.  **Environment validation** --- checks for a CUDA-enabled GPU and
    installs missing Python dependencies without installing or replacing
    PyTorch.
2.  **PDF ingestion** --- uploads the three required PDF documents.
3.  **Text extraction** --- extracts embedded PDF text page by page
    using PyMuPDF.
4.  **Dataset construction** --- creates page-level JSONL and CSV
    records with source metadata.
5.  **Train/validation split** --- splits at page level before
    tokenization, groups exact duplicate page text to reduce leakage,
    and uses a fixed random seed.
6.  **Tokenization and chunking** --- tokenizes each page independently
    and creates causal-LM training chunks without crossing page
    boundaries.
7.  **Quantized model loading** --- loads the pretrained Qwen checkpoint
    in 4-bit NF4.
8.  **QLoRA adaptation** --- attaches trainable LoRA adapters to
    selected attention and MLP projection modules.
9.  **Training and evaluation** --- trains for the configured epochs and
    evaluates next-token prediction loss on held-out pages.
10. **Artifact export** --- saves the adapter, tokenizer, training
    configuration, metrics, manifests, and a ZIP package.

## Model and training configuration

  -----------------------------------------------------------------------------------------------------------------------
  Component                           Configuration
  ----------------------------------- -----------------------------------------------------------------------------------
  Base model                          [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

  Training method                     Continued causal language-model pretraining

  Parameter-efficient method          QLoRA with PEFT LoRA adapters

  Quantization                        4-bit NF4 with double quantization

  Compute dtype                       FP16

  LoRA rank (`r`)                     16

  LoRA alpha                          32

  LoRA dropout                        0.05

  LoRA target modules                 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

  Epochs                              3

  Per-device batch size               1

  Gradient accumulation               8

  Learning rate                       2e-4

  Maximum sequence length             512 tokens

  Validation fraction                 10% of eligible pages

  Random seed                         42

  Intended runtime                    Google Colab with CUDA GPU; notebook is tuned for Tesla T4
  -----------------------------------------------------------------------------------------------------------------------

These are the notebook defaults and can be changed in the relevant
configuration cells. Actual runtime, GPU, training duration, dataset
sizes, and loss values depend on the run and are recorded by the
notebook when generated.

## Repository contents

At minimum, this repository should contain:

``` text
PalgharGPT-QLoRA/
├── README.md
├── requirements.txt
├── .gitignore
└── PalgharGPT_QLoRA_Three_Documents.ipynb
```

After running the notebook, it generates artifacts in its Colab working
directory, including:

``` text
PalgharGPT/
├── original_pdfs/
├── dataset/
│   ├── palghar_master_dataset.jsonl
│   ├── palghar_master_dataset.csv
│   ├── dataset_metadata.json
│   ├── page_split_manifest.json
│   └── token_chunk_manifest.json
└── output/
    ├── PalgharGPT_LoRA/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer files
    │   ├── training_configuration.json
    │   ├── evaluation_metrics.json
    │   ├── training_manifest.json
    │   └── trainer_log_history.json
    └── PalgharGPT_QLoRA_No_RAG.zip
```

The exact set of generated files may vary depending on the run. The
adapter package contains the LoRA adapter and tokenizer, **not a
standalone full copy of the base model**. The original Qwen base
checkpoint is required to load the adapter.

## Run the notebook

`requirements.txt` lists the Python packages used by the notebook. In Google Colab, the notebook checks for and installs missing packages automatically; PyTorch is intentionally left to the CUDA runtime image.

### Requirements

-   A Google account with access to Google Colab.
-   A CUDA-enabled GPU runtime. The notebook is configured with Tesla
    T4-oriented settings.
-   The three source PDFs listed above.
-   Internet access in Colab to download the base model and Python
    dependencies.

### Steps

1.  Open the `.ipynb` notebook in Google Colab.
2.  Select **Runtime → Change runtime type → T4 GPU** (or another
    compatible CUDA GPU).
3.  Run the notebook from top to bottom using **Runtime → Run all**.
4.  When the upload dialog appears, select all three required PDF files.
5.  Allow the notebook to extract text, prepare the dataset, load the
    base model, train the LoRA adapter, and evaluate it.
6.  Download and safely store the generated
    `PalgharGPT_QLoRA_No_RAG.zip` before the Colab session ends.

The notebook checks that the required PDFs are present and validates
their PDF signatures. Browser-renamed copies with suffixes such as `(1)`
are handled.

## Reload the trained adapter

The notebook includes a reload example. The essential loading pattern
is:

``` python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = "/path/to/PalgharGPT_LoRA"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()
```

Replace `adapter_dir` with the actual path to the extracted adapter
folder. A compatible CUDA environment and the required dependencies must
be available.

## Evaluation

The notebook records training loss, validation loss, runtime, peak GPU
memory use, and train/validation page and chunk counts. It also runs a
small generation smoke test using example Palghar-related prompts.

**Validation loss is not a factual-accuracy score.** It measures
next-token prediction on held-out pages from the same small document
collection. It does not establish that the model gives correct
statistics, dates, citations, or complete answers. A separate factual
benchmark with source-page references is needed before making
reliability claims.

## Limitations and responsible use

-   **No RAG or source retrieval:** answers are generated from model
    parameters; source passages are not fetched at query time.
-   **No OCR:** PyMuPDF extracts embedded/selectable text only. Scanned
    or image-only pages may produce little or no text and are reported
    by the notebook.
-   **Small corpus:** three documents are a limited training source and
    cannot represent every topic or the latest situation in Palghar.
-   **Factual uncertainty:** the model may omit, mix up, or generate
    inaccurate statistics, dates, administrative details, and
    recommendations.
-   **Reference periods matter:** publication year is not necessarily
    the year to which a statistic applies. Check the relevant table or
    source page.
-   **Not an official source:** PalgharGPT is an experimental research
    project, not a government system or an authoritative source of
    district information.
-   **Colab sessions are temporary:** download the adapter ZIP and any
    required dataset artifacts before disconnecting.

Do not use model-generated output as the sole basis for public policy,
financial decisions, health decisions, or other high-impact decisions.
Verify important claims against the original source documents.

## Future improvements

-   Add more authoritative, clearly dated Palghar district documents.
-   Introduce OCR as a separate, auditable preprocessing stage for
    scanned pages.
-   Build a page-cited factual evaluation benchmark.
-   Compare the adapted model against the untouched base model.
-   Explore a separate RAG pipeline when traceable source-grounded
    answers are required.
-   Publish reproducible training metrics and dataset-quality reports.

## Acknowledgements

-   **Base model:**
    [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
-   **Model tooling:** Hugging Face Transformers, PEFT, Datasets, and
    bitsandbytes.
-   **PDF text extraction:** PyMuPDF.
-   **Training environment:** Google Colab / CUDA GPU.

## License and attribution

The base model is subject to its own model license and terms. The source
PDFs may have separate copyright, licensing, or usage conditions. Before
redistributing model adapters, extracted datasets, or source documents,
review the applicable terms and provide appropriate attribution. Add a
repository license only after confirming that you have the rights to
license all materials you are publishing.

------------------------------------------------------------------------

**Project:** PalgharGPT\
**Method:** QLoRA-based continued causal language-model pretraining\
**Scope:** Palghar district, Maharashtra, India\
**Retrieval:** No RAG