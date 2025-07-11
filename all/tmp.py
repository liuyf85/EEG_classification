import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(TransformerBinaryClassifier, self).__init__()
        # 初始化一个 BERT 配置
        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=8,  # 你可以根据需要调整
            num_hidden_layers=4,    # 你可以根据需要调整
            intermediate_size=hidden_size * 4,
            max_position_embeddings=2100,
            vocab_size=30522  # 这个值可以随意设定，因为我们不会用到词汇表
        )
        
        self.bert = BertModel(config)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, x):
        # BERT expects input ids, token type ids, and attention masks
        # Since we don't have a tokenized input, we'll use dummy values for token type ids and attention masks
        attention_mask = torch.ones(x.shape[:2]).to(x.device)
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)
        
        # 方法1：使用 [CLS] 标记的表示
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits_cls = self.classifier(cls_output)
        
        # 方法2：平均池化所有时间步的输出
        mean_output = torch.mean(outputs.last_hidden_state, dim=1)
        logits_mean = self.classifier(mean_output)
        
        # 方法3：最大池化所有时间步的输出
        max_output, _ = torch.max(outputs.last_hidden_state, dim=1)
        logits_max = self.classifier(max_output)
        
        # 返回不同方法的输出，实际应用中可以选择其中一个
        return logits_cls, logits_mean, logits_max

# 创建一个 (90, 2100, 19) 的输入张量
input_tensor = torch.randn(90, 2100, 19)

# 初始化模型
model = TransformerBinaryClassifier(hidden_size=19, num_labels=2)

# 进行前向传播
logits_cls, logits_mean, logits_max = model(input_tensor)

# 输出张量的形状
print(logits_cls.shape)  # 应该是 (90, 2)
print(logits_mean.shape)  # 应该是 (90, 2)
print(logits_max.shape)  # 应该是 (90, 2)