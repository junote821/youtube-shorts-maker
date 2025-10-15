import base64
from google.genai import types
from openai import OpenAI
from google.adk.tools.tool_context import ToolContext

client = OpenAI()


async def generate_images(tool_context: ToolContext):
    # 1. state에 접근하여 prompt를 받아오는 것
    prompt_builder_output = tool_context.state.get("prompt_builder_output")
    optimized_prompts = prompt_builder_output.get("optimized_prompts")
    
    # 2. Artifacts가 이미 생성되었는지 확인 -> 생성한 이미지를 또 생성하는 일이 없도록
    existing_artifacts = await tool_context.list_artifacts()
    
    # 도구가 출력물을 갖도록 설정도 가능
    generated_images = [] # 생성된 이미지 정보를 담을 것
    
    # 3. optimized prompt에 있는 각 프롬프트를 실행
    for prompt in optimized_prompts:
        # optimized_prompt에서 scene_id와 enhanced_prompt 추출
        scene_id = prompt.get("scene_id")
        enhanced_prompt = prompt.get("enhanced_prompt")
        
        # 파일이 존재하는지 확인할 때 사용할 파일명 생성 -> artifacts에 존재하는지 확인
        filename = f"scene_{scene_id}_image.jpeg"
        if filename in existing_artifacts:
            generated_images.append(
                {
                    "scene_id": scene_id,
                    "prompt": enhanced_prompt[:100], # 프롬프트 앞의 100자
                    "filename": filename
                }
            )
            continue # file이 존재하니까 다음 루프로
        
        # OpenAI와 대화 -> 필요한 건 다 있으니.. 이미지를 생성(파일이름도 생성, 프롬프트도 존재)
        image = client.images.generate( # 생성되는 이미지는 base64로 인코딩되어 출력
            model="gpt-image-1",
            prompt=enhanced_prompt,
            n=1,
            quality="auto",
            moderation="low",
            output_format="jpeg",
            background="opaque",
            size="1024x1536", 
        )
        
        # 디코딩 작업
        image_bytes = base64.b64decode(image.data[0].b64_json) # 첫번째 값만 있으면 됨
        
        # byte로 artifacts로 생성
        artifact = types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=image_bytes
            )
        )
        # artifact 저장
        await tool_context.save_artifact(filename=filename, artifact=artifact)
        
        generated_images.append(
            {
                "scene_id": scene_id,
                "prompt": enhanced_prompt[:100], # 프롬프트 앞의 100자
                "filename": filename
            }
        )
        
        return {
            "total_images": len(generated_images),
            "generated_images": generated_images,
            "status": "complete"
        }