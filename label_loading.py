import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def filter_training_labels(train_df, ext_df, cfg, random_state=None):
    """
    Filter training labels by selecting the 5-second segment with highest confidence for each file.
    
    For each entry in train_df:
    1. Find all corresponding 5-second samples in ext_df
    2. Select the one with highest probability for the primary label
    3. Keep secondary labels only if they have sufficient probability in the chosen 5s segment
    """
    if random_state is None: 
        random_state = cfg.seed
    
    # Group ext_df by samplename (filename without extension)
    ext_df['samplename'] = ext_df['row_id'].apply(lambda x: x.split('_')[0])
    
    # Create a lookup for faster access
    ext_grouped = ext_df.groupby('samplename')
    
    return_list = []
    
    for _, train_row in tqdm(train_df.iterrows(), desc="Filtering training data", total=len(train_df)):
        # Extract samplename from the full filename
        samplename = train_row['filename'].split('/')[-1].split('.')[0]
        
        # Get primary label and secondary labels
        primary_label = train_row['primary_label']
        secondary_labels = train_row['secondary_labels']
        
        # Process secondary labels
        if secondary_labels not in [[''], None, np.nan, [], "['']", "[]"]:
            if isinstance(secondary_labels, str):
                try:
                    secondary_labels = eval(secondary_labels)
                except:
                    secondary_labels = [secondary_labels]
        else:
            secondary_labels = []
                
        # Skip if no corresponding entries in ext_df
        if samplename not in ext_grouped.groups:
            print(f"Warning: {samplename} not found in external dataframe, instead applying exisiting labels")
            timestamp = "5" # first 5 second always exist
            retained_secondaries = secondary_labels
        else:
            # Get all 5-second segments for this sample
            segments = ext_grouped.get_group(samplename)
                
            # Find segment with highest probability for primary label
            segments_with_probs = segments.copy()
            segments_with_probs['primary_prob'] = segments_with_probs[primary_label]
            best_segment = segments_with_probs.loc[segments_with_probs['primary_prob'].idxmax()]
            
            # Skip if confidence for primary label is too low
            if best_segment['primary_prob'] < cfg.train_label_confidence:
                continue
                
            # Filter secondary labels based on their confidence in the best segment
            retained_secondaries = []
            for sec_label in secondary_labels:
                if sec_label in segments.columns[1:] and best_segment[sec_label] >= cfg.secondary_label_confidence:
                    retained_secondaries.append(sec_label)
            
            # Extract timestamp from row_id
            timestamp = best_segment['row_id'].split('_')[1]
        
        # Add the filtered entry to our results
        return_list.append({
            "samplename": samplename + "_" + timestamp,
            "primary_label": primary_label,
            "secondary_labels": str(retained_secondaries),
            "filename": samplename + ".ogg",
            "filepath": os.path.join(cfg.train_datadir, samplename + ".ogg"),
            "timestamp": timestamp,
        })
    
    result_df = pd.DataFrame(return_list)

    # shuffle the final DataFrame
    result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"Returning {len(result_df)} processed filtered labels")
    return result_df


def load_pseudolabels(df, cfg, seed=None):
    """
    Load pseudolabels and sample to ensure balanced class representation.
    Returns a maximum of cfg.max_pseudolabels samples with balanced label distribution.
    """
    random_seed = seed if seed is not None else cfg.seed
    np.random.seed(random_seed)
    
    if cfg.debug:
        df = df.sample(min(1000, len(df)), random_state=random_seed).reset_index(drop=True)
    
    label_cols = df.columns[1:]
    
    # Step 1: Find samples where at least one prediction passes the confidence threshold
    # Calculate row sums first to filter out rows with no valid predictions
    df_filtered = df.copy()
    row_sums = df_filtered[label_cols].sum(axis=1)
    df_filtered = df_filtered[row_sums > 0].reset_index(drop=True)
    
    # Normalize probabilities to sum to 1 for each row
    if cfg.normalize_labels:
        df_filtered[label_cols] = df_filtered[label_cols].div(df_filtered[label_cols].sum(axis=1), axis=0)
    
    # Create mask for values that pass threshold
    mask = df_filtered[label_cols] >= cfg.pseudolabel_confidence_threshold
    
    # Get indices and values of samples with predictions above threshold using the mask
    valid_samples = {}  # {row_id: [labels sorted by confidence]}
    
    for idx, row in df_filtered.iterrows():
        row_id = row['row_id']
        # Use the mask to get valid labels for this row
        valid_mask = mask.iloc[idx]
        valid_label_cols = label_cols[valid_mask]
        
        if len(valid_label_cols) > 0:
            # Get confidences for valid labels and sort them
            confidences = row[valid_label_cols]
            # Create (confidence, label) pairs and sort by confidence
            label_conf_pairs = [(label, conf) for label, conf in zip(valid_label_cols, confidences)]
            label_conf_pairs.sort(key=lambda x: x[1], reverse=True)
            # Extract just the sorted labels
            valid_samples[row_id] = [label for label, _ in label_conf_pairs]
    
    print(f"Found {len(valid_samples)} samples with confidence > {cfg.pseudolabel_confidence_threshold}")
    
    # Step 2: Apply stratified sampling if enabled
    if cfg.stratified_pseudolabels and len(valid_samples) > cfg.max_pseudolabels:
        print("Stratified sampling of pseudolabels")
        
        # Create label-to-samples mapping
        label_to_samples = {}
        for sample_id, labels in valid_samples.items():
            for label in labels:
                if label not in label_to_samples:
                    label_to_samples[label] = []
                label_to_samples[label].append(sample_id)
        
        # Calculate target samples per label for balanced distribution
        num_unique_labels = len(label_to_samples)
        target_per_label = cfg.max_pseudolabels // num_unique_labels
        
        print(f"Found {num_unique_labels} unique bird species in pseudolabels")
        print(f"Target ~{target_per_label} samples per species to stay under {cfg.max_pseudolabels} total")

        # Sample entries for each label
        selected_samples = set()
        
        for label in label_to_samples.keys():
            samples = label_to_samples[label]
            # Shuffle samples with deterministic but different seed per label
            samples_seed = random_seed + hash(label) % 10000
            np.random.seed(samples_seed)
            random.shuffle(samples)
            
            # Take at most target_per_label samples
            selected = samples[:target_per_label]
            selected_samples.update(selected)
        
        # Filter valid_samples to only include selected samples
        valid_samples = {k: v for k, v in valid_samples.items() if k in selected_samples}
        print(f"Sampled {len(valid_samples)} pseudolabeled samples")
    else:
        print(f"Using randomly selected {cfg.max_pseudolabels} pseudolabels that pass confidence threshold")
    
    rows_to_keep = random.sample(list(valid_samples.keys()), min(cfg.max_pseudolabels, len(valid_samples)))

    # Step 3: Convert to training-compatible DataFrame format
    label_df = pd.DataFrame({
        'samplename': rows_to_keep,
        'filename': [f"{'_'.join(x.split('_')[:3])}.ogg" for x in rows_to_keep],
        'timestamp': [int(x.split('_')[3]) for x in rows_to_keep],
        'primary_label': [valid_samples[x][0] for x in rows_to_keep],
        'secondary_labels': [valid_samples[x][1:] for x in rows_to_keep]
    })
    
    # Add filepath column 
    label_df['filepath'] = label_df['filename'].apply(lambda x: os.path.join(cfg.train_soundscapes, x))
    
    return label_df


def load_soft_pseudolabels(df, cfg, seed=None):
    """
    Load soft pseudolabels.
    """
    # Can use different seeds e.g. for different folds
    random_seed = seed if seed is not None else cfg.seed

    print(f"Found {len(df)} samples in the pseudolabels")
    if cfg.debug:
        df = df.sample(min(1000, len(df)), random_state=random_seed).reset_index(drop=True)
    
    label_col = df.columns[1:]
    
    # Normalize probabilities to sum to 1 for each row
    if cfg.normalize_labels:
        df[label_col] = df[label_col].div(df[label_col].sum(axis=1), axis=0)
        
    # Sample to restrict to max_pseudolabels
    df = df.sample(min(cfg.max_pseudolabels, len(df)), random_state=random_seed).reset_index(drop=True)
    row_ids = df["row_id"]

    # Convert to training-compatible DataFrame format
    label_df = pd.DataFrame({
        'samplename': row_ids,
        'filename': [f"{'_'.join(x.split('_')[:3])}.ogg" for x in row_ids],
        'primary_label' : ["soft"] * len(row_ids),
        'secondary_labels': [df.loc[df["row_id"]==x].to_numpy()[0, 1:].astype(np.float32) for x in row_ids]
    })
    label_df['filepath'] = label_df['filename'].apply(lambda x: os.path.join(cfg.train_soundscapes, x))
    print(f"Loaded {len(label_df)} soft pseudolabels")

    return label_df