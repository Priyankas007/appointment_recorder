# Medical Appointment Recorder

A Flask web application for processing medical PDFs and providing live transcription with speaker diarization.

## Features

### 📄 PDF Medical Record Processing
- Upload multiple PDF medical records
- Extract text using pdfplumber
- Generate AI-powered health summaries using OpenAI GPT models
- Fallback to placeholder summaries when no API key is available

### 🎤 Live Transcription with Diarization
- Real-time audio recording from microphone
- Speaker diarization to identify different speakers
- Live transcription display with confidence scores
- Session management and statistics

### 🎵 Audio Upload & Playback
- Upload various audio formats (mp3, mp4, m4a, wav, aac, ogg)
- Built-in audio/video player
- File management with unique naming

## Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd appointment_recorder
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements_minimal.txt
   ```

4. **Set up environment variables (optional):**
   ```bash
   # For OpenAI AI summaries
   export OPENAI_API_KEY="your-openai-api-key"
   
   # For AWS Transcribe (real transcription)
   export AWS_ACCESS_KEY_ID="your-aws-access-key"
   export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
   export AWS_REGION="us-east-1"  # or your preferred region
   ```

5. **Run the application:**
   ```bash
   PORT=5001 python3 app.py
   ```

6. **Access the application:**
   Open your browser and go to `http://127.0.0.1:5001`

## Usage

### PDF Processing
1. Click "Choose Files" in the PDF section
2. Select one or more medical PDF files
3. Click "Generate Summary"
4. View the AI-generated health summary

### Live Transcription
1. Click "🎤 Start Recording" to begin
2. Allow microphone access when prompted
3. Speak into your microphone
4. View real-time transcription with speaker identification
5. Click "⏹️ Stop Recording" to end the session

### Audio Upload
1. Click "Choose Files" in the Audio Upload section
2. Select audio/video files
3. Click "Upload & Play"
4. Use the built-in player to listen to your files

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for AI summaries
- `AWS_ACCESS_KEY_ID`: AWS access key for transcription
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for transcription
- `AWS_REGION`: AWS region (default: us-east-1)
- `PORT`: Server port (default: 5000)
- `MAX_UPLOAD_MB`: Maximum upload size in MB (default: 100)

### AWS Transcribe Setup
To enable real AWS Transcribe streaming with diarization:

1. **Create an AWS account** and get your access keys
2. **Set up IAM permissions** for Amazon Transcribe:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Sid": "transcribe-streaming-policy",
               "Effect": "Allow",
               "Action": "transcribe:StartStreamTranscription",
               "Resource": "*"
           }
       ]
   }
   ```
3. **Set environment variables** with your AWS credentials
4. **Restart the application**

## Technical Details

### Architecture
- **Backend**: Flask web framework
- **PDF Processing**: pdfplumber for text extraction
- **AI Summaries**: OpenAI GPT models
- **Transcription**: AWS Transcribe with speaker diarization
- **Frontend**: Vanilla JavaScript with modern CSS

### API Endpoints
- `POST /summarize`: Process PDF files and generate summaries
- `POST /upload-audio`: Upload audio files
- `GET /media/audio/<filename>`: Serve uploaded audio files
- `POST /transcribe/start`: Start transcription session
- `POST /transcribe/stream/<session_id>`: Stream audio data
- `GET /transcribe/events/<session_id>`: Get transcription events (SSE)
- `POST /transcribe/end/<session_id>`: End transcription session

### File Structure
```
appointment_recorder/
├── app.py                 # Main Flask application
├── requirements_minimal.txt # Python dependencies
├── README.md             # This file
├── uploads_audio/        # Uploaded audio files (created automatically)
└── venv/                 # Virtual environment (created during setup)
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   PORT=5001 python3 app.py
   ```

2. **Module not found errors:**
   ```bash
   source venv/bin/activate
   pip install -r requirements_minimal.txt
   ```

3. **AWS Transcribe not working:**
   - Verify AWS credentials are set correctly
   - Check IAM permissions for Transcribe
   - Ensure AWS region is correct

4. **Microphone access denied:**
   - Allow microphone access in your browser
   - Check browser permissions settings

### Demo Mode
When AWS credentials are not configured, the transcription feature runs in demo mode with mock data to demonstrate the interface and functionality.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.
