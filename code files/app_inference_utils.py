import os
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "backend", "saved_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("BASE_DIR :", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)

# -------------------------------------------------
# LOAD SAVED FILES
# -------------------------------------------------
config = joblib.load(os.path.join(MODEL_DIR, "config.pkl"))

drug2id = joblib.load(os.path.join(MODEL_DIR, "drug2id.pkl"))
ind2id = joblib.load(os.path.join(MODEL_DIR, "ind2id.pkl"))
se2id = joblib.load(os.path.join(MODEL_DIR, "se2id.pkl"))

id2drug = joblib.load(os.path.join(MODEL_DIR, "id2drug.pkl"))
id2ind = joblib.load(os.path.join(MODEL_DIR, "id2ind.pkl"))
id2se = joblib.load(os.path.join(MODEL_DIR, "id2se.pkl"))

edge_index = torch.load(os.path.join(MODEL_DIR, "edge_index.pt"), map_location=DEVICE)
edge_type = torch.load(os.path.join(MODEL_DIR, "edge_type.pt"), map_location=DEVICE)

# -------------------------------------------------
# MODEL CLASSES
# -------------------------------------------------
class RGCNEncoder(nn.Module):
    def __init__(self, num_nodes, num_relations, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)

        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(RGCNConv(emb_dim, hidden_dim, num_relations))
        else:
            self.convs.append(RGCNConv(emb_dim, hidden_dim, num_relations))
            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        self.dropout = dropout

    def forward(self, edge_index, edge_type):
        x = self.node_emb.weight

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class MLPTripleScorer(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 5, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1)
        )

    def forward(self, z_d, z_i, z_s):
        x = torch.cat(
            [z_d, z_i, z_s, z_d * z_s, z_i * z_s],
            dim=-1
        )
        return self.net(x).squeeze(-1)


class ADRRGCNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = RGCNEncoder(
            num_nodes=config["num_nodes"],
            num_relations=config["num_relations"],
            emb_dim=config["emb_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
        self.scorer = MLPTripleScorer(
            hidden_dim=config["hidden_dim"],
            mlp_hidden=config["mlp_hidden"],
            dropout=config["dropout"],
        )

    def encode(self, edge_index, edge_type):
        return self.encoder(edge_index, edge_type)

    def score(self, z, d, i, s):
        d = torch.tensor([d], dtype=torch.long, device=z.device)
        i = torch.tensor([i], dtype=torch.long, device=z.device)
        s = torch.tensor([s], dtype=torch.long, device=z.device)

        z_d = z[d + config["drug_offset"]]
        z_i = z[i + config["ind_offset"]]
        z_s = z[s + config["se_offset"]]

        logit = self.scorer(z_d, z_i, z_s)
        return torch.sigmoid(logit).item()


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = ADRRGCNModel(config).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "rgcn_model.pt"), map_location=DEVICE))
model.eval()

print("✅ RGCN model loaded successfully!")
print("✅ edge_index shape:", tuple(edge_index.shape))
print("✅ edge_type shape :", tuple(edge_type.shape))


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def clean_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def probability_to_confidence(prob, threshold):
    if prob >= max(threshold, 0.80):
        return 0.95
    elif prob >= max(threshold * 0.7, 0.50):
        return 0.75
    return 0.55


# -------------------------------------------------
# MAIN PREDICTION FUNCTION
# -------------------------------------------------
def predict_adverse_reactions(drug_name, indication_name, side_effects_list):
    predictions = []

    drug_name = clean_text(drug_name)
    indication_name = clean_text(indication_name)

    if not side_effects_list:
        return predictions

    if drug_name not in drug2id:
        print(f"❌ Drug not found: {drug_name}")
        return predictions

    if indication_name not in ind2id:
        print(f"❌ Indication not found: {indication_name}")
        return predictions

    d_id = drug2id[drug_name]
    i_id = ind2id[indication_name]
    threshold = float(config.get("best_threshold", 0.5))

    seen = set()

    try:
        with torch.no_grad():
            z = model.encode(edge_index, edge_type)

            for side_effect in side_effects_list:
                se_name = clean_text(side_effect)

                if not se_name:
                    continue

                if se_name in seen:
                    continue
                seen.add(se_name)

                if se_name not in se2id:
                    continue

                s_id = se2id[se_name]

                prob = model.score(z, d_id, i_id, s_id)
                prob = max(0.0, min(1.0, float(prob)))

                confidence = probability_to_confidence(prob, threshold)

                predictions.append((
                    se_name,
                    round(prob, 4),
                    round(confidence, 4)
                ))

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return []

    predictions.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return predictions