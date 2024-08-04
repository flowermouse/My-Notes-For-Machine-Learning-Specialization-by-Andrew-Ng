DOC = notebook

# 定义编译器和选项
LATEX = xelatex
LATEX_OPTS = -shell-escape

# 默认目标
all: $(DOC).pdf

# 生成 PDF 文件
$(DOC).pdf: $(DOC).tex $(wildcard chapter*.tex) preface.tex
	$(LATEX) $(LATEX_OPTS) $(DOC).tex
	$(LATEX) $(LATEX_OPTS) $(DOC).tex

# 清理生成的文件
clean:
	del /Q $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).lof $(DOC).lot $(DOC).idx \
	$(DOC).ind $(DOC).ilg $(DOC).synctex.gz $(DOC).code $(DOC).ex $(DOC).thm $(DOC).dfn \
	$(wildcard chapter*.aux) preface.aux preface.log preface.out preface.toc preface.lof \
	preface.lot preface.idx preface.ind preface.ilg preface.synctex.gz

# 清理所有生成的文件，包括 PDF
distclean: clean
	del /Q $(DOC).pdf