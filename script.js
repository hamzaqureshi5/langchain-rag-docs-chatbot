function sendMessage() {
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');
    const fileInput = document.getElementById('fileInput');

    if (userInput.value.trim() !== '') {
        appendMessage(userInput.value, 'right', '#007bff', 'white');
        userInput.value = ''; // Clear input after sending
    }

    if (fileInput.files.length > 0) {
        const fileName = fileInput.files[0].name;
        appendMessage(`Uploaded file: ${fileName}`, 'left', '#f0f0f0', 'black');
        fileInput.value = ''; // Clear file input after sending
    }

    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
}

function appendMessage(text, align, bgColor, textColor) {
    const messageDiv = document.createElement('div');
    messageDiv.textContent = text;
    messageDiv.style.background = bgColor;
    messageDiv.style.color = textColor;
    messageDiv.style.padding = '10px';
    messageDiv.style.margin = '5px';
    messageDiv.style.borderRadius = '5px';
    messageDiv.style.textAlign = align;
    document.getElementById('chatBox').appendChild(messageDiv);
}
