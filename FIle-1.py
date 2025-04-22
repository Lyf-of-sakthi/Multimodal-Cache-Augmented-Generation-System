import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import re

# Initialize the NVIDIA client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def extract_video_id(youtube_url):
    # Extract video ID from various YouTube URL formats
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def summarize_with_llm(transcript):
    prompt = f"""
    Please summarize the following podcast transcript in a detailed and comprehensive manner.
    Include key points, main arguments, and important insights.
    The summary should be well-structured and easy to follow.
    
    Transcript:
    {transcript[:15000]}  # Limiting to 15k chars to avoid token limits
    
    Summary:
    """
    
    response = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates detailed summaries of podcast content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    )
    
    return response.choices[0].message.content

def main():
    st.title("🎙️ YouTube Podcast Summarizer")
    st.write("Enter a YouTube podcast link to get a detailed summary of its content.")
    
    youtube_url = st.text_input("YouTube Podcast URL:")
    
    if st.button("Summarize"):
        if youtube_url:
            with st.spinner("Processing..."):
                # Extract video ID
                video_id = extract_video_id(youtube_url)
                if not video_id:
                    st.error("Invalid YouTube URL. Please enter a valid YouTube link.")
                    return
                
                # Get transcript
                transcript = get_transcript(video_id)
                if not transcript:
                    st.error("Could not retrieve transcript. The video might not have captions.")
                    return
                
                # Summarize with LLM
                try:
                    summary = summarize_with_llm(transcript)
                    
                    # Display results
                    st.subheader("Podcast Summary")
                    st.write(summary)
                    
                    # Show transcript (collapsed)
                    with st.expander("View Full Transcript"):
                        st.text(transcript[:10000] + "... [truncated]")  # Showing first 10k chars
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
        else:
            st.warning("Please enter a YouTube URL")

if __name__ == "__main__":
    main()