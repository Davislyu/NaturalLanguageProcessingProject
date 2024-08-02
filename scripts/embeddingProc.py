import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os


def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_data(filename):
    df = pd.read_csv(filename)
    return df["text"].tolist()


def sanitize_text_data(texts):
    sanitized_texts = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            sanitized_texts.append(text)
        else:
            sanitized_texts.append("N/A")
    return sanitized_texts


def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    model.to(device)
    model.eval()
    all_embeddings = []

    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Processing texts"):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]
        batch_texts = sanitize_text_data(batch_texts)

        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=300,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                mean_embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(mean_embeddings.cpu())

        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    else:
        print("No valid embeddings generated.")
        return torch.empty(0)


def generate_embeddings(input_file, output_file):
    device = check_device()

    tokenizer_ct = AutoTokenizer.from_pretrained(
        "digitalepidemiologylab/covid-twitter-bert"
    )
    model_ct = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")

    tokenizer_sbert = AutoTokenizer.from_pretrained(
        "sentence-transformers/stsb-roberta-large"
    )
    model_sbert = AutoModel.from_pretrained("sentence-transformers/stsb-roberta-large")

    texts = load_data(input_file)
    texts = sanitize_text_data(texts)

    ctbert_embeddings = get_embeddings(
        texts, model_ct, tokenizer_ct, device, batch_size=16
    )
    sbert_embeddings = get_embeddings(
        texts, model_sbert, tokenizer_sbert, device, batch_size=16
    )

    combined_embeddings = torch.cat((ctbert_embeddings, sbert_embeddings), dim=1)

    if combined_embeddings.shape[0] > 0:
        # Save the combined embeddings to Parquet format
        df_embeddings = pd.DataFrame(combined_embeddings.numpy())
        df_embeddings.to_parquet(output_file.replace(".xlsx", ".parquet"), index=False)
        print(
            f"Final embeddings saved to '{output_file.replace('.xlsx', '.parquet')}'."
        )
    else:
        print("No embeddings were saved due to errors in processing.")


def main():
    input_file = os.path.join("./trainingSetRaw.csv")
    output_file = os.path.join(
        "./final_training.parquet"
    )  # Changed extension to .parquet
    generate_embeddings(input_file, output_file)


if __name__ == "__main__":
    main()
