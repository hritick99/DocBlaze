from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, file_type: str):
    if file_type == "pdf":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    elif file_type in ["csv", "xlsx", "xls"]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

    return splitter.split_text(text)
