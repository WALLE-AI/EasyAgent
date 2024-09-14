import asyncio
import os
import aiohttp

from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from load_env import load_env
load_env

async def main():
  async with aiohttp.ClientSession() as session:
    # Use Daily as a real-time media transport (WebRTC)
    transport = DailyTransport(
      room_url="https://starvlm.daily.co/StarVLM",
      token=os.getenv("DAILY_KEY"),
      bot_name="StarVLM",
      params=DailyParams(audio_out_enabled=True))

    # Use Cartesia for Text-to-Speech
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_TTS_KEY"),
        voice_id="easy-agent"
      )

    # Simple pipeline that will process text to speech and output the result
    pipeline = Pipeline([tts, transport.output()])

    # Create Pipecat processor that can run one or more pipelines tasks
    runner = PipelineRunner()

    # Assign the task callable to run the pipeline
    task = PipelineTask(pipeline)

    # Register an event handler to play audio when a
    # participant joins the transport WebRTC session
    @transport.event_handler("on_participant_joined")
    async def on_new_participant_joined(transport, participant):
      participant_name = participant["info"]["userName"] or ''
      # Queue a TextFrame that will get spoken by the TTS service (Cartesia)
      await task.queue_frames([TextFrame(f"Hello there, {participant_name}!"), EndFrame()])

    # Run the pipeline task
    await runner.run(task)

if __name__ == "__main__":
  asyncio.run(main())