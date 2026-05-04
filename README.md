# An Analysis of Social Engineering Attacks on AI Detection Model Security

This project investigates how phishing email detection systems do against adversarial attacks, particularly those targeting transformer-based models like BERT and large language models such as ChatGPT.
# A Comparison of Data Sanitization Methods
## Objective
The objective of this project is to demonstrate the effectiveness of various *data Sanitization* methods. 

Deleted files will be inspected for artifacts using third party tools such as [Eric Zimmerman Tools](https://ericzimmerman.github.io/#!index.md) as well as built in tools such as Windows' Registry Editor.

## File Deletion
File deletion will replicate common user deletion methods e.g. right clicking and sending to the Recycle Bin. Other deletion scenarios cover using [Window's Cipher](https://learn.microsoft.com/en-us/troubleshoot/windows-server/certificates-and-public-key-infrastructure-pki/use-cipher-to-overwrite-deleted-data) cipher /w:D to overwrite deleted data, using the command prompt del command, and the Poweshell `Remove-Item'. 
