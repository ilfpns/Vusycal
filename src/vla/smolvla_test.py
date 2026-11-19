
import torch

from lerobot.configs import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"

# 1. Device 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# 2. 모델 로드
print("Loading SmolVLA...")

policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
policy = policy.to(device).eval()

# 3. 전처리 / 후처리 파이프라인
preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    MODEL_ID,
    preprocessor_overrides={
        "device_processor": {"device": str(device)}
    },
)

# 4. 더미 입력 생성
# 실제 모델 설정에서 입력 Feature 구조를 가져옴
frame = {}
image_added = False

for name, feature in policy.config.input_features.items():
    if feature.type == FeatureType.VISUAL:
        # 테스트용 RGB 이미지 하나 생성
        if not image_added:
            frame[name] = torch.rand(feature.shape)
            image_added = True

    else:
        # 로봇 관절 상태 등의 더미 데이터
        frame[name] = torch.zeros(feature.shape)

# 자연어 명령
frame["task"] = "pick up the cube"

print("\nInput:")
for key, value in frame.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {tuple(value.shape)}")
    else:
        print(f"{key}: {value}")

# 5. 전처리
batch = preprocessor(frame)

# 6. 추론
print("\nRunning inference...")

with torch.inference_mode():
    actions = policy.predict_action_chunk(batch)

# 7. 출력 확인
print("\nOutput:")
print(f"Action shape: {tuple(actions.shape)}")
print(f"Action dtype: {actions.dtype}")
print(f"Action device: {actions.device}")

print("\nInference completed!")

