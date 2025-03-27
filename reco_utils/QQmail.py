import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from typing import Optional, Dict, Any
import json
from pprint import pformat


class ExperimentEmailSender:
    """
    A class to send experiment results via email.

    Attributes:
        from_addr (str): Sender's email address
        password (str): Email account password or authorization code
        smtp_server (str): SMTP server address
        smtp_port (int): SMTP server port
    """

    def __init__(self, from_addr: str = '', password: str = '',
                 smtp_server: str = 'smtp.qq.com', smtp_port: int = 465):
        self.from_addr = from_addr
        self.password = password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def format_dict(self, data: Dict[str, Any], indent: int = 4) -> str:
        """
        Format dictionary into a readable HTML string.

        Args:
            data: Dictionary containing experiment results
            indent: Number of spaces for indentation

        Returns:
            str: Formatted HTML string
        """
        # 使用pformat获取格式化的字符串
        formatted_text = pformat(data, indent=indent, width=120)
        # 转换为HTML格式
        html_text = formatted_text.replace('\n', '<br>')
        html_text = html_text.replace(' ', '&nbsp;')
        return f'<pre style="font-family: monospace;">{html_text}</pre>'

    def send_experiment_results(self, to_addr: str, results: Dict[str, Any],
                                experiment_name: str = "实验结果") -> bool:
        """
        Send experiment results via email.

        Args:
            to_addr: Recipient's email address
            results: Dictionary containing experiment results
            experiment_name: Name of the experiment for the email subject

        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        try:
            # Create message container
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = Header(to_addr)
            msg['Subject'] = Header(f"{experiment_name}完成通知", 'utf-8')

            # Format the results
            html_content = f"""
            <html>
            <body>
                <h2>{experiment_name}已完成</h2>
                <div>
                    {self.format_dict(results)}
                </div>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_content, 'html', 'utf-8'))

            # Create SMTP connection and send
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as smtp:
                smtp.login(self.from_addr, self.password)
                smtp.sendmail(self.from_addr, to_addr, msg.as_string())
            return True

        except smtplib.SMTPException as e:
            return False