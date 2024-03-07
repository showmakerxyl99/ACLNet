import torch
import torch.nn.functional as F

# Example usage
# dbcl = DBCL_Module(feature_dim=512)
# feature_vectors and labels from ABFO module and DataLoader respectively
# dbcl.update_dicts(labels, feature_vectors)
# Calculate contrastive loss
# loss = dbcl.compute_contrastive_loss(support_features, support_labels)

class DBCL_Module:
    def __init__(self, feature_dim, max_vectors_per_class=10):
        self.category_dict = {}
        self.prototype_dict = {}
        self.feature_dim = feature_dim
        self.max_vectors_per_class = max_vectors_per_class

    def update_dicts(self, labels, feature_vectors):
        for label, vector in zip(labels, feature_vectors):
            label = label.item()
            if label not in self.category_dict:
                self.category_dict[label] = []
            self.category_dict[label].append(vector)
            if len(self.category_dict[label]) > self.max_vectors_per_class:
                self.category_dict[label].pop(0)
            self.prototype_dict[label] = torch.mean(torch.stack(self.category_dict[label]), dim=0)



    def compute_contrastive_loss(self, support_features, support_labels):
        loss = 0
        for feature, label in zip(support_features, support_labels):
            # Compute loss with prototype dictionary
            positive_proto = self.prototype_dict[label.item()]
            negative_protos = [self.prototype_dict[lbl] for lbl in self.prototype_dict if lbl != label.item()]
            proto_loss = self.compute_loss(feature, positive_proto, negative_protos)
    
            # Compute loss with category dictionary
            positive_category = torch.mean(torch.stack(self.category_dict[label.item()]), dim=0)
            negative_categories = [torch.mean(torch.stack(self.category_dict[lbl]), dim=0) for lbl in self.category_dict if
                                   lbl != label.item()]
            category_loss = self.compute_loss(feature, positive_category, negative_categories)
    
            # Combine losses with alpha hyperparameter
            loss += self.alpha * proto_loss + (1 - self.alpha) * category_loss
    
        return loss / len(support_features)

    def compute_loss(self, feature, positive, negatives):
        positive_dist = F.pairwise_distance(feature.unsqueeze(0), positive.unsqueeze(0))
        negative_dists = torch.stack([F.pairwise_distance(feature.unsqueeze(0), neg.unsqueeze(0)) for neg in negatives])
        return positive_dist - torch.logsumexp(-negative_dists, dim=0)