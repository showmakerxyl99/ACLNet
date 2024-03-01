import torch
import torch.nn as nn
import torch.nn.functional as F


class IIMA(nn.Module):
    def __init__(self, config):
        super(IIMA, self).__init__()
        self.config = config
        self.feature_dimension = 640
        self.num_classes = config.num_class
        self.temperature = config.temperature
        self.shot = config.shot
        # Encoder and transformation layers remain the same

        self.co_attention_map = nn.Sequential(
            nn.Conv2d(25, 5, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 25, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # Classifier remains the same

    def compute_integration(self, feature_q, feature_s):
        # Feature normalization and transformation code remains the same

        # Compute cosine similarity
        feature_mul = torch.einsum('qncij,qnckl->qnijkl', feature_s_c, feature_q_c)

        # Compute co-attention
        co_attention_s, co_attention_q = self.compute_co_attention(feature_mul)

        # Aggregate attended features
        attended_s = self.aggregate_attended_features(co_attention_s, feature_s, mode='support')
        attended_q = self.aggregate_attended_features(co_attention_q, feature_q, mode='query')

        # Compute final similarity score
        similarity_score = F.cosine_similarity(attended_s, attended_q, dim=-1) / self.temperature

        return similarity_score if not self.training else (similarity_score, self.classifier(attended_q))

    def compute_co_attention(self, feature_similarity):
        # Assuming feature_similarity has dimensions [num_qry, way, H_s, W_s, H_q, W_q]
        # Normalize to get co-attention scores
        s_attention = F.softmax(feature_similarity.mean(dim=[4, 5]), dim=-1)
        q_attention = F.softmax(feature_similarity.mean(dim=[2, 3]), dim=-1)
        return s_attention, q_attention

    def aggregate_attended_features(self, co_attention, original_features, mode):
        # Reshape co-attention to match original feature dimensions
        reshaped_attention = co_attention.view(*co_attention.shape[:2], *original_features.shape[-2:])

        # Apply co-attention on original features
        if mode == 'support':
            # Aggregate for support features
            attended_features = (reshaped_attention.unsqueeze(2) * original_features.unsqueeze(0)).mean(dim=1)
        else:
            # Aggregate for query features
            attended_features = (reshaped_attention.unsqueeze(2) * original_features.unsqueeze(1)).mean(dim=1)

        return attended_features

