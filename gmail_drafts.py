import os.path
import base64
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these SCOPES, delete the token.json file.
SCOPES = ['https://www.googleapis.com/auth/gmail.compose']

def gmail_authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If no valid credentials, ask the user to log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def create_message(sender, to, subject, message_text, is_html=True):
    subtype = 'html' if is_html else 'plain'
    
    message = MIMEText(message_text, _subtype=subtype)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'message': {'raw': raw_message}}


def create_draft(service, user_id, message_body):
    try:
        draft = service.users().drafts().create(userId=user_id, body=message_body).execute()
        print(f"Draft id: {draft['id']}\nDraft message: {draft['message']}")
        return draft
    except Exception as error:
        print(f'An error occurred: {error}')
        return None

def main():
  service = gmail_authenticate()
  message_body = create_message(
    sender='abhijeethkollarapu@gmail.com',
    to='abhijeeth1200@example.com',
    subject='This is a draft email2',
    message_text='Hello, this is a draft!2'
  )
  create_draft(service, 'me', message_body)


if __name__ == '__main__':
    main()
