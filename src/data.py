import pandas as pd
import numpy as np 
import ast

def create_clean_sample_map(df, df_ent) : 
    entity_map = {}

    for _, row in df_ent.iterrows():
        ticker = row["Symbol"]
        
        # assume variants is list or string
        variants = row["Entity"].split(sep = ', ')
        
        for v in variants:
            entity_map[v.lower()] = ticker

    bad_rows = []

    for idx, val in df["Decisions"].items():
        try:
            ast.literal_eval(val)
        except Exception as e:
            bad_rows.append({
                "index": idx,
                "value": val,
                "error": str(e)
            })

    bad_df = pd.DataFrame(bad_rows)

    print(f"{len(bad_df)} problematic rows")
    print(bad_df.head())

    bad_indices = bad_df['index'].to_list()

    samples = []

    for _, row in df.drop(index=bad_indices).iterrows():
        headline = row["Title"]
        entities = ast.literal_eval(row["Decisions"])

        for target_entity, sentiment in entities.items():
            sample = {
                "headline": headline,
                "target": target_entity,
                "entities": list(entities.keys()),
                "label": sentiment
            }
            samples.append(sample)

    # 3615
    entities = ["Sebi", "Osian's Art Fund"]
    samples.append({"headline": "Osian's Art Fund approaches Tribunal against Sebi order", "target": "Sebi", "entities": entities, "label": "neutral"})
    samples.append({"headline": "Osian's Art Fund approaches Tribunal against Sebi order", "target": "Osian's Art Fund", "entities": entities, "label": "neutral"})

    # 3783
    entities = ["Tata Steel UK", "Moody's"]
    samples.append({"headline": "Moody's downgrades Tata Steel UK's rating by one notch", "target": "Tata Steel UK", "entities": entities, "label": "negative"})
    samples.append({"headline": "Moody's downgrades Tata Steel UK's rating by one notch", "target": "Moody's", "entities": entities, "label": "neutral"})

    # 4909
    entities = ["Sun Pharma", "Dr Reddy's Labs"]
    samples.append({"headline": "Pharma companies like Sun Pharma, Dr Reddy's Labs likely to do well", "target": "Sun Pharma", "entities": entities, "label": "positive"})
    samples.append({"headline": "Pharma companies like Sun Pharma, Dr Reddy's Labs likely to do well", "target": "Dr Reddy's Labs", "entities": entities, "label": "positive"})

    # 5014
    entities = ["Sun Pharma", "Dr. Reddy's"]
    samples.append({"headline": "Sun Pharma and Dr. Reddy's are top pharma stocks: Mitesh Thacker", "target": "Sun Pharma", "entities": entities, "label": "positive"})
    samples.append({"headline": "Sun Pharma and Dr. Reddy's are top pharma stocks: Mitesh Thacker", "target": "Dr. Reddy's", "entities": entities, "label": "positive"})

    # 5321
    entities = ["IOC", "BPCL", "HPCL", "Moody's"]
    samples.append({"headline": "IOC, BPCL, HPCL slipped on Moody's view", "target": "IOC", "entities": entities, "label": "negative"})
    samples.append({"headline": "IOC, BPCL, HPCL slipped on Moody's view", "target": "BPCL", "entities": entities, "label": "negative"})
    samples.append({"headline": "IOC, BPCL, HPCL slipped on Moody's view", "target": "HPCL", "entities": entities, "label": "negative"})
    samples.append({"headline": "IOC, BPCL, HPCL slipped on Moody's view", "target": "Moody's", "entities": entities, "label": "neutral"})

    # 7226
    entities = ["BP", "Britain's FTSE"]
    samples.append({"headline": "BP lifts Britain's FTSE towards 4-1/2 month high", "target": "BP", "entities": entities, "label": "positive"})
    samples.append({"headline": "BP lifts Britain's FTSE towards 4-1/2 month high", "target": "Britain's FTSE", "entities": entities, "label": "positive"})

    # 8167
    entities = ["Nikkei", "BOJ", "Moody's"]
    samples.append({"headline": "Nikkei hits 7-yr high, hopes of BOJ buying stocks offsets Moody's", "target": "Nikkei", "entities": entities, "label": "positive"})
    samples.append({"headline": "Nikkei hits 7-yr high, hopes of BOJ buying stocks offsets Moody's", "target": "BOJ", "entities": entities, "label": "neutral"})
    samples.append({"headline": "Nikkei hits 7-yr high, hopes of BOJ buying stocks offsets Moody's", "target": "Moody's", "entities": entities, "label": "neutral"})

    # 8202
    entities = ["Samsung", "Barron's"]
    samples.append({"headline": "Samsung shares could gain as much as 50 per cent: Barron's", "target": "Samsung", "entities": entities, "label": "positive"})
    samples.append({"headline": "Samsung shares could gain as much as 50 per cent: Barron's", "target": "Barron's", "entities": entities, "label": "neutral"})

    # 9624
    entities = ["Citic Securities", "China's central bank"]
    samples.append({"headline": "Citic Securities urges China's central bank to cut rates, reserve requirements", "target": "Citic Securities", "entities": entities, "label": "negative"})
    samples.append({"headline": "Citic Securities urges China's central bank to cut rates, reserve requirements", "target": "China's central bank", "entities": entities, "label": "neutral"})

    # 9824
    entities = ["Diageo", "BAE", "UK's FTSE"]
    samples.append({"headline": "Diageo and BAE fall as UK's FTSE backs off from record high", "target": "Diageo", "entities": entities, "label": "negative"})
    samples.append({"headline": "Diageo and BAE fall as UK's FTSE backs off from record high", "target": "BAE", "entities": entities, "label": "negative"})
    samples.append({"headline": "Diageo and BAE fall as UK's FTSE backs off from record high", "target": "UK's FTSE", "entities": entities, "label": "negative"})

    # 10269
    entities = ["Sun Pharma", "Dr Reddy's Laboratories"]
    samples.append({"headline": "Compliance worries hit shares of Sun Pharma and  Dr Reddy's Laboratories", "target": "Sun Pharma", "entities": entities, "label": "negative"})
    samples.append({"headline": "Compliance worries hit shares of Sun Pharma and  Dr Reddy's Laboratories", "target": "Dr Reddy's Laboratories", "entities": entities, "label": "negative"})

    # 10355
    entities = ["Axis Bank", "HDFC Bank", "ICICI Bank", "Moody's"]
    samples.append({"headline": "Moody's reaffirms ratings of Axis Bank, HDFC Bank and ICICI Bank", "target": "Axis Bank", "entities": entities, "label": "positive"})
    samples.append({"headline": "Moody's reaffirms ratings of Axis Bank, HDFC Bank and ICICI Bank", "target": "HDFC Bank", "entities": entities, "label": "positive"})
    samples.append({"headline": "Moody's reaffirms ratings of Axis Bank, HDFC Bank and ICICI Bank", "target": "ICICI Bank", "entities": entities, "label": "positive"})
    samples.append({"headline": "Moody's reaffirms ratings of Axis Bank, HDFC Bank and ICICI Bank", "target": "Moody's", "entities": entities, "label": "neutral"})

    # 10621
    entities = ["Ranbaxy", "Dr. Reddy's", "Cadila", "HSBC"]
    samples.append({"headline": "Ranbaxy, Dr. Reddy's, Cadila look attractive: Jitendra Sriram, HSBC", "target": "Ranbaxy", "entities": entities, "label": "positive"})
    samples.append({"headline": "Ranbaxy, Dr. Reddy's, Cadila look attractive: Jitendra Sriram, HSBC", "target": "Dr. Reddy's", "entities": entities, "label": "positive"})
    samples.append({"headline": "Ranbaxy, Dr. Reddy's, Cadila look attractive: Jitendra Sriram, HSBC", "target": "Cadila", "entities": entities, "label": "positive"})
    samples.append({"headline": "Ranbaxy, Dr. Reddy's, Cadila look attractive: Jitendra Sriram, HSBC", "target": "HSBC", "entities": entities, "label": "neutral"})

    return samples 

def transform_text(headline, target_entity, all_entities):
    
    text = headline
    text = text.replace(target_entity, "TARGET")
    for e in all_entities:
        if e != target_entity:
            text = text.replace(e, "OTHER")
    return text

def replace_entities_with_target_other(samples) : 
    
    for s in samples:
        s["headline"] = transform_text(s["headline"], s["target"], s["entities"])

    return samples
