#!pip install -q docling pytesseract sentence-transformers faiss-cpu
#!sudo apt-get install -y -qq tesseract-ocr tesseract-ocr-por tesseract-ocr-eng

import os
import torch
import faiss
import numpy as np
import json
import statistics
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ============================================================
# CONFIGURAÇÃO GLOBAL
# ============================================================
BASE_DIR   = Path("/content/drive/MyDrive/SB100")
OVERLAP    = 0
CHUNK_LIST = [512, 800, 1024]

# ============================================================
# LISTA DE PDFs PARA TESTAR
# ============================================================
pdfs = [
    {
        "nome": "Arantes(2020)-Microbiome",
        "caminho": "/content/drive/MyDrive/SB100/data/Citrus/Application of biostimulant consortium_bioinsumos.pdf",
        "language": ["eng"],
        "perguntas": [
            "What are the main components of the biostimulant consortium used in this study?",
            "At what stage after planting was the Sucrosin applied via foliar spray?",
            "How did the application of the biostimulant consortium affect the height and stem diameter of sugarcane plants compared to the control?",
            "What environmental condition during the experiment negatively impacted the growth of sugarcane plants between 4 to 6 months after planting?",
            "Explain how humic acid and mycorrhizal fungi contribute to nutrient availability and plant stress tolerance according to the study.",
            "Discuss the potential mechanisms by which the biostimulant consortium maintains sugarcane vegetative growth under drought stress conditions."
        ]
    },
    # Adicione mais PDFs aqui seguindo o mesmo padrão
]

# ============================================================
# SALVA RESULTADOS ANTERIORES
# ============================================================
print("📦 Verificando resultados anteriores do Zambrosi...")
for letra in ["A", "B", "C"]:
    src = Path(f"/content/drive/MyDrive/SB100/Teste/Teste_Soil{letra}.json")
    dst = Path(f"/content/drive/MyDrive/SB100/Teste/Teste_Soil{letra}_salvo.json")
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"   ✅ Zambrosi {letra} salvo em {dst.name}")
    elif dst.exists():
        print(f"   ℹ️  Zambrosi {letra} já salvo anteriormente")
    else:
        print(f"   ⚠️  Zambrosi {letra} não encontrado")

# ============================================================
# CONFIGURAÇÃO DO TESSERACT
# ============================================================
candidates = [
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
    "/usr/share/tesseract-ocr/tessdata",
    "/usr/share/tessdata",
]

tessdata = next((p for p in candidates if os.path.isdir(p)), None)
if tessdata is None:
    tessdata = "/usr/share/tesseract-ocr/5/tessdata"
    os.makedirs(tessdata, exist_ok=True)

os.environ["TESSDATA_PREFIX"] = tessdata
print(f"\nTESSDATA_PREFIX → {tessdata}")

needed  = ["osd.traineddata", "por.traineddata", "eng.traineddata"]
missing = [f for f in needed if not os.path.isfile(os.path.join(tessdata, f))]

if missing:
    print(f"⚠️  Instalando arquivos faltando: {missing}")
    os.system("apt-get install -y -qq tesseract-ocr-por tesseract-ocr-eng 2>&1")
    tessdata = next((p for p in candidates if os.path.isdir(p)), tessdata)
    os.environ["TESSDATA_PREFIX"] = tessdata
    print("✅ Tesseract configurado!")
else:
    print("✅ Tesseract OK!")

# ============================================================
# FUNÇÕES
# ============================================================
def clean_text(text):
    text = text.replace("glyph<c=3,font=/CIDFont+F5>", " ")
    text = text.replace("glyph<c=3,font=/CIDFont+F8>", " ")
    text = text.replace("&gt;", "").replace("&lt;", "")
    return text

def pdf_to_markdown(file_path, ocr_lang, md_dir, data_dir):
    base_stem = Path(file_path).stem
    md_path   = data_dir / f"{base_stem}.md"
    if md_path.exists():
        print(f"ℹ️  Markdown já existe, reutilizando!")
        return str(md_path)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options = TesseractCliOcrOptions(lang=ocr_lang, force_full_page_ocr=True)
    pipeline_options.generate_picture_images = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    print(f"🔄 Convertendo PDF...")
    print(f"   Tamanho: {os.path.getsize(file_path)/(1024*1024):.2f} MB")
    result = converter.convert(file_path).document
    for text in getattr(result, "texts", []):
        text.orig = clean_text(getattr(text, "orig", ""))
    for table in getattr(result, "tables", []):
        for cell in getattr(table.data, "table_cells", []):
            cell.text = clean_text(getattr(cell, "text", ""))
    picture_counter = 0
    for element, _ in result.iterate_items():
        if isinstance(element, PictureItem):
            picture_counter += 1
            img_path = md_dir / f"{base_stem}-picture-{picture_counter}.png"
            with img_path.open("wb") as fp:
                element.get_image(result).save(fp, "PNG")
    print(f"   {picture_counter} imagens salvas")
    result_md = result.export_to_markdown()
    with md_path.open("w", encoding="utf-8") as f:
        f.write(result_md)
    print(f"✅ Markdown salvo em: {md_path}")
    return str(md_path)

def fazer_chunks(md_path, chunk_size):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    content = [p.strip() for p in content]
    content = [p for p in content if p.replace('-','').replace('|','').replace(' ','').strip()]
    chunks        = []
    current_chunk = ""
    for i, paragraph in enumerate(content):
        if i == 0:
            current_chunk += paragraph
            continue
        if paragraph.startswith("| ") and paragraph.endswith(" |"):
            current_chunk += "\n" + paragraph
            continue
        if len(current_chunk) + len(paragraph) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += "\n" + paragraph
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def avaliar_tamanho_chunks(chunks, label):
    tamanhos  = [len(c) for c in chunks]
    ideal     = sum(1 for t in tamanhos if 300 <= t <= 1024)
    score_tam = (ideal / len(chunks)) * 100
    print(f"\n{'='*55}")
    print(f"📏 AVALIAÇÃO DE TAMANHO — {label}")
    print(f"{'='*55}")
    print(f"   Total chunks  : {len(chunks)}")
    print(f"   Menor         : {min(tamanhos)} chars")
    print(f"   Maior         : {max(tamanhos)} chars")
    print(f"   Média         : {statistics.mean(tamanhos):.0f} chars")
    print(f"   Mediana       : {statistics.median(tamanhos):.0f} chars")
    print(f"   Desvio padrão : {statistics.stdev(tamanhos):.0f} chars")
    print(f"\n   Distribuição:")
    print(f"   < 200  : {sum(1 for t in tamanhos if t < 200):>4} ⚠️")
    print(f"   200-400: {sum(1 for t in tamanhos if 200 <= t < 400):>4}")
    print(f"   400-600: {sum(1 for t in tamanhos if 400 <= t < 600):>4} ✅")
    print(f"   600-800: {sum(1 for t in tamanhos if 600 <= t < 800):>4} ✅")
    print(f"   800-1024:{sum(1 for t in tamanhos if 800 <= t < 1024):>4} ✅")
    print(f"   > 1024 : {sum(1 for t in tamanhos if t > 1024):>4} ⚠️")
    print(f"\n   ⭐ Score tamanho: {score_tam:.1f}% ({ideal}/{len(chunks)} no ideal)")
    return score_tam

def criar_indice_faiss(chunks):
    print(f"🔄 Gerando embeddings para {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=16)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"✅ Índice FAISS criado com {index.ntotal} vetores!")
    return index

def buscar_faiss(pergunta, index, chunks):
    vetor = model.encode([pergunta])
    vetor = np.array(vetor).astype('float32')
    faiss.normalize_L2(vetor)
    scores, indices = index.search(vetor, 1)
    return {"score": float(scores[0][0]), "chunk": chunks[indices[0][0]], "chars": len(chunks[indices[0][0]])}

# ============================================================
# EMBEDDING — carregado uma única vez
# ============================================================
print("\n🔄 Carregando modelo de embedding (Qwen3 multilingual)...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device='cuda')
print("✅ Modelo carregado!")

# ============================================================
# LOOP PRINCIPAL — itera sobre cada PDF
# ============================================================
todos_resultados_global = {}

for pdf_cfg in pdfs:
    nome      = pdf_cfg["nome"]
    PDF_TESTE = pdf_cfg["caminho"]
    OCR_LANG  = pdf_cfg["language"]
    perguntas = pdf_cfg["perguntas"]

    # Diretórios isolados por PDF (slug do nome)
    slug     = nome.replace(" ", "_").replace("/", "-")
    MD_DIR   = BASE_DIR / "testes" / slug / "md_images"
    DATA_DIR = BASE_DIR / "testes" / slug / "data_md"
    MD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'█'*60}")
    print(f"█ PDF: {nome}")
    print(f"█ Idioma: {OCR_LANG} | Perguntas: {len(perguntas)}")
    print(f"{'█'*60}")

    # Converte PDF uma vez por documento
    md_path = pdf_to_markdown(PDF_TESTE, OCR_LANG, MD_DIR, DATA_DIR)

    todos_resultados = []

    for chunk_size in CHUNK_LIST:
        label = f"chunk_{chunk_size}"
        print(f"\n{'#'*60}")
        print(f"# EXECUTANDO: {nome} | chunk_size = {chunk_size}")
        print(f"{'#'*60}")

        chunks    = fazer_chunks(md_path, chunk_size)
        score_tam = avaliar_tamanho_chunks(chunks, label)
        index     = criar_indice_faiss(chunks)

        print(f"\n{'='*60}")
        print(f"🔍 RESULTADOS — {nome} | chunk_size={chunk_size}")
        print(f"   Modelo: Qwen3-Embedding-0.6B (multilingual)")
        print(f"{'='*60}")

        scores = []
        for pergunta in perguntas:
            r         = buscar_faiss(pergunta, index, chunks)
            relevante = "✅" if r['score'] > 0.55 else "⚠️" if r['score'] > 0.45 else "❌"
            scores.append(r['score'])
            print(f"\n❓ {pergunta}")
            print(f"   Score : {r['score']:.4f} {relevante} | Chars: {r['chars']}")
            print(f"   Texto : {r['chunk'][:200]}...")
            print("-"*60)

        media_score = sum(scores) / len(scores)
        print(f"\n📈 MÉDIA  : {media_score:.4f}")
        print(f"   Tamanho: {score_tam:.1f}%")
        print(f"   ✅ bons : {sum(1 for s in scores if s > 0.55)}/{len(scores)}")
        print(f"   ⚠️ ok   : {sum(1 for s in scores if 0.45 <= s <= 0.55)}/{len(scores)}")
        print(f"   ❌ ruins: {sum(1 for s in scores if s < 0.45)}/{len(scores)}")

        resultado = {
            "chunk_size": chunk_size, "label": label,
            "total_chunks": len(chunks),
            "menor_chunk": min(len(c) for c in chunks),
            "maior_chunk": max(len(c) for c in chunks),
            "media_chunk": sum(len(c) for c in chunks) // len(chunks),
            "score_tamanho": score_tam,
            "scores": scores,
            "media_score": media_score,
            "md_nome": Path(md_path).stem
        }
        todos_resultados.append(resultado)

    # Salva resultado individual por PDF
    output_path = f"/content/resultado_{slug}.json"
    with open(output_path, "w") as f:
        json.dump({
            "pdf": nome,
            "modelo": "Qwen3-Embedding-0.6B",
            "idioma": OCR_LANG,
            "resultados": todos_resultados
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Salvo em: {output_path}")

    # Ranking por PDF
    print(f"\n🏆 RANKING — {nome}:")
    ranking = sorted(todos_resultados, key=lambda x: x['media_score'], reverse=True)
    for i, r in enumerate(ranking, 1):
        print(f"   {i}º chunk_{r['chunk_size']:>4} → score={r['media_score']:.4f} | tam={r['score_tamanho']:.1f}%")

    todos_resultados_global[nome] = todos_resultados

# ============================================================
# SALVA RESULTADO CONSOLIDADO DE TODOS OS PDFs
# ============================================================
with open("/content/resultado_consolidado.json", "w") as f:
    json.dump(todos_resultados_global, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"✅ TODOS OS PDFs PROCESSADOS!")
print(f"   PDFs testados: {[p['nome'] for p in pdfs]}")
print(f"   Consolidado em: /content/resultado_consolidado.json")