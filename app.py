from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

def should_stop_generation(output, stop_patterns):
    for pattern in stop_patterns:
        if pattern in output:
            return True
    return False

async def generate(
    prompt, history, temperature=0.7, max_new_tokens=1024, top_p=0.95, repetition_penalty=1.0,
    stop_patterns=None, max_loops=5
):
    if stop_patterns is None:
        stop_patterns = ["\n\n", ".", "The end", "Thank you"]

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)

    output = ""
    loop_count = 0
    while loop_count < max_loops:
        try:
            stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
            async for response in stream:
                output += response.token.text
                yield output
                if should_stop_generation(output, stop_patterns):
                    return  # Stop if end pattern is detected

            loop_count += 1  # Increment loop count to avoid infinite loops

            # If the text isn't complete, use the last segment as a new prompt
            formatted_prompt = format_prompt(output.split("\n")[-1], history)
        
        except Exception as e:
            print(f"Error during streaming: {e}")
            break

    # Non-streaming fallback
    try:
        response = client.text_generation(formatted_prompt, **generate_kwargs, stream=False, return_full_text=False)
        output = response  # Use response as a string if streaming failed
        if not should_stop_generation(output, stop_patterns):
            output += " [Additional text required to complete the response.]"
        yield output
    except Exception as e:
        print(f"Error during non-streaming generation: {e}")

mychatbot = gr.Chatbot(
    avatar_images=["./user.png", "./botm.png"], bubble_full_width=False, show_label=False, show_copy_button=True, likeable=True,
)

demo = gr.ChatInterface(fn=generate, 
                        chatbot=mychatbot,
                        title="Mixtral 8x7b AI Chatbot By wifix199",
                        retry_btn=None,
                        undo_btn=None
                       )

demo.queue().launch(server_name="10.172.2.37", server_port=5002, show_api=False, share=True)
