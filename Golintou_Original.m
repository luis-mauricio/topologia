%Golintou - Programa para Análise Topológica de Lineamentos com Interface

format compact;
clear;
clc;

fprintf('*** Golintou - Programa para Análise Topológica de Lineamentos com Interface ***\n\n');

Golintop();

function Golintop

%interface para entrada de dados
fig = figure('Name', 'Golintou', 'NumberTitle', 'off', 'Position', [000, 050, 1600, 740]);

%dados iniciais
data.vizinhanca = 0.25;
data.arquivo = 'Frattopo.xyz';
data.discretizacao = 12;

%painel para entrada de dados
panel = uipanel(fig, 'Title', 'Entrada de Dados', 'Position', [0.01, 0.01, 0.20, 0.98]);

%adicionando campos para edicao de dados
fields = fieldnames(data);
numFields = numel(fields);
inputs = cell(numFields, 1);
for i = 1:numFields
    uicontrol(panel, 'Style', 'text', 'String', fields{i}, 'HorizontalAlignment', 'left', ...
        'Position', [10, 650-30*i, 150, 20]);
    inputs{i} = uicontrol(panel, 'Style', 'edit', 'String', mat2str(data.(fields{i})), ...
        'Position', [80, 650-30*i, 200, 20]);
end

%botao para executar os calculos
uicontrol(panel, 'Style', 'pushbutton', 'String', 'Executar', ...
    'Position', [120, 70, 100, 30], 'Callback', @executarCalculos);

%eixos para graficos
ax = cell(4,2);
for i = 1:4
    for j = 1:2
        ax{i,j} = subplot(4, 2, (i-1)*2+j, 'Parent', fig);
    end
end

    function [] = executarCalculos(~, ~)
        %limpar os graficos existentes
        for m = 1:4
            for n = 1:2
                cla(ax{m,n});
            end
        end
        %atualizar dados com as entradas do usuario
        for k = 1:numFields
            data.(fields{k}) = eval(inputs{k}.String);
        end

        factor = data.vizinhanca;
        arquivo = data.arquivo;
        lados = data.discretizacao;

        %entrada dos dados

        fid = fopen(arquivo);
        file = textscan(fid,'%f%f%f%f');
        fclose(fid);

        %calculos com fator de aproximacao

        xi = round(file{1}/factor)*factor;
        yi = round(file{2}/factor)*factor;
        xf = round(file{3}/factor)*factor;
        yf = round(file{4}/factor)*factor;

        %deteccao das interseccoes ou toques dos lineamentos

        XY1 = [xi yi xf yf];
        XY2 = XY1;

        [n_rows_1,n_cols_1] = size(XY1);
        [n_rows_2,n_cols_2] = size(XY2);

        X1 = repmat(XY1(:,1),1,n_rows_2);
        X2 = repmat(XY1(:,3),1,n_rows_2);
        Y1 = repmat(XY1(:,2),1,n_rows_2);
        Y2 = repmat(XY1(:,4),1,n_rows_2);

        XY2 = XY2';

        X3 = repmat(XY2(1,:),n_rows_1,1);
        X4 = repmat(XY2(3,:),n_rows_1,1);
        Y3 = repmat(XY2(2,:),n_rows_1,1);
        Y4 = repmat(XY2(4,:),n_rows_1,1);

        X4_X3 = (X4-X3);
        Y1_Y3 = (Y1-Y3);
        Y4_Y3 = (Y4-Y3);
        X1_X3 = (X1-X3);
        X2_X1 = (X2-X1);
        Y2_Y1 = (Y2-Y1);

        numerator_a = X4_X3 .* Y1_Y3 - Y4_Y3 .* X1_X3;
        numerator_b = X2_X1 .* Y1_Y3 - Y2_Y1 .* X1_X3;
        denominator = Y4_Y3 .* X2_X1 - X4_X3 .* Y2_Y1;

        u_a = numerator_a./denominator;
        u_b = numerator_b./denominator;

        INT_X = X1+X2_X1.*u_a;
        INT_Y = Y1+Y2_Y1.*u_a;
        INT_B = (u_a >= 0) & (u_a <= 1) & (u_b >= 0) & (u_b <= 1);
        PAR_B = denominator == 0;
        COINC_B = (numerator_a == 0 & numerator_b == 0 & PAR_B);

        out.intAdjacencyMatrix = INT_B;
        out.intMatrixX = INT_X .* INT_B;
        out.intMatrixY = INT_Y .* INT_B;
        out.intNormalizedDistance1To2 = u_a;
        out.intNormalizedDistance2To1 = u_b;
        out.parAdjacencyMatrix = PAR_B;
        out.coincAdjacencyMatrix= COINC_B;

        %determinacao dos lineamentos que nao se intersectam ou se tocam

        flagni = 1;
        counter = 0;
        for n=1:length(xi)
            if out.intAdjacencyMatrix(n,:)==0
                counter = counter+1;
                xi2(counter) = xi(n);
                yi2(counter) = yi(n);
                xf2(counter) = xf(n);
                yf2(counter) = yf(n);
            end
        end
        if counter==0
            flagni = 0;
            fprintf('todos lineamentos têm pelo menos uma intersecção ou se tocam!\n\n');
        end

        %determinacao dos lineamentos que se intersectam ou se tocam

        flagin = 1;
        counter = 0;
        for n=1:length(xi)
            if sum(out.intAdjacencyMatrix(n,:))>=1
                counter = counter+1;
                xi3(counter) = xi(n);
                yi3(counter) = yi(n);
                xf3(counter) = xf(n);
                yf3(counter) = yf(n);
            end
        end
        if counter==0
            flagin = 0;
            fprintf('não há lineamentos que se intersectam ou se tocam!\n\n');
        end

        %determinacao dos pontos de interseccao ou toque

        xp = 0;
        yp = 0;
        counter = 0;
        for n=1:length(xi)
            for m=n:length(xi)
                if out.intAdjacencyMatrix(n,m)==1
                    counter = counter+1;
                    xp(counter) = out.intMatrixX(n,m);
                    yp(counter) = out.intMatrixY(n,m);
                end
            end
        end

        %divisao dos segmentos provenientes das interseccoes ou toques

        counter = 0;
        for n=1:length(xi)
            if sum(out.intAdjacencyMatrix(n,:))>=1
                xpi = xi(n);
                ypi = yi(n);
                for m=1:length(xi)
                    if out.intAdjacencyMatrix(n,m)==1
                        counter = counter+1;
                        xi4(counter) = xpi;
                        yi4(counter) = ypi;
                        xf4(counter) = out.intMatrixX(n,m);
                        yf4(counter) = out.intMatrixY(n,m);
                        xpi = xf4(counter);
                        ypi = yf4(counter);
                    end
                end
                xi4(counter+1) = xpi;
                yi4(counter+1) = ypi;
                xf4(counter+1) = xf(n);
                yf4(counter+1) = yf(n);
                counter = length(xi4);
            end
        end

        %calculo do raio de desconexao

        if flagin==1
            for n=1:length(xi4)
                L(n) = sqrt((xf4(n)-xi4(n))^2+(yf4(n)-yi4(n))^2);
            end
            counter = 0;
            for n=1:length(xi4)
                if L(n)~=0
                    counter = counter+1;
                    Lt(counter) = L(n);
                end
            end
            r = min(Lt)/2;%adequar aos dados!
        end

        %desconexao dos lineamentos que se intersectam ou se tocam e discriminacao Y

        counter = 0;
        n_typey = 0;
        M = zeros(1,2);
        xi5 = 0; yi5 = 0; xf5 = 0; yf5 = 0;
        for n=1:length(xi)
            if sum(out.intAdjacencyMatrix(n,:))>=1
                alpha = abs(atand((yf(n)-yi(n))/(xf(n)-xi(n))));
                xpi = xi(n);
                ypi = yi(n);
                for m=1:length(xi)
                    flagpass = 0;
                    if out.intAdjacencyMatrix(n,m)==1
                        flaginiti = 0;
                        flagfinal = 0;
                        counter = counter+1;
                        xi5(counter) = xpi;
                        yi5(counter) = ypi;
                        if xi(n)==out.intMatrixX(n,m) & yi(n)==out.intMatrixY(n,m)
                            flaginiti = 1;
                            n_typey = n_typey+1;
                            x6y(n_typey) = xi(n);
                            y6y(n_typey) = yi(n);
                        end
                        if xf(n)==out.intMatrixX(n,m) & yf(n)==out.intMatrixY(n,m)
                            flagfinal = 1;
                            n_typey = n_typey+1;
                            x6y(n_typey) = xf(n);
                            y6y(n_typey) = yf(n);
                        end
                        if xpi<=out.intMatrixX(n,m) & ypi<=out.intMatrixY(n,m)
                            if flaginiti==0
                                xf5(counter) = out.intMatrixX(n,m)-cosd(alpha)*r;
                                yf5(counter) = out.intMatrixY(n,m)-sind(alpha)*r;
                                xpi = out.intMatrixX(n,m)+cosd(alpha)*r;
                                ypi = out.intMatrixY(n,m)+sind(alpha)*r;
                                if flagfinal==1
                                    xf5(counter) = out.intMatrixX(n,m);
                                    yf5(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            else
                                if flaginiti==1 & flagpass==0
                                    flagpass = 1;
                                    xf5(counter) = out.intMatrixX(n,m);
                                    yf5(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            end
                            xpi_end = xpi;
                            ypi_end = ypi;
                        end
                        if xpi>=out.intMatrixX(n,m) & ypi>=out.intMatrixY(n,m)
                            if flaginiti==0
                                xf5(counter) = out.intMatrixX(n,m)+cosd(alpha)*r;
                                yf5(counter) = out.intMatrixY(n,m)+sind(alpha)*r;
                                xpi = out.intMatrixX(n,m)-cosd(alpha)*r;
                                ypi = out.intMatrixY(n,m)-sind(alpha)*r;
                                if flagfinal==1
                                    xf5(counter) = out.intMatrixX(n,m);
                                    yf5(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            else
                                if flaginiti==1 & flagpass==0
                                    flagpass = 1;
                                    xf5(counter) = out.intMatrixX(n,m);
                                    yf5(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            end
                            xpi_end = xpi;
                            ypi_end = ypi;
                        end
                        if xpi<=out.intMatrixX(n,m) & ypi>=out.intMatrixY(n,m)
                            if flaginiti==0
                                xf5(counter) = out.intMatrixX(n,m)-cosd(alpha)*r;
                                yf5(counter) = out.intMatrixY(n,m)+sind(alpha)*r;
                                xpi = out.intMatrixX(n,m)+cosd(alpha)*r;
                                ypi = out.intMatrixY(n,m)-sind(alpha)*r;
                                if flagfinal==1
                                    xf5(counter) = out.intMatrixX(n,m);
                                    yf5(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            else
                                if flaginiti==1 & flagpass==0
                                    flagpass = 1;
                                    xf5(counter) = out.intMatrixX(n,m);
                                    yf5(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            end
                            xpi_end = xpi;
                            ypi_end = ypi;
                        end
                        if xpi>=out.intMatrixX(n,m) & ypi<=out.intMatrixY(n,m)
                            if flaginiti==0
                                xf5(counter) = out.intMatrixX(n,m)+cosd(alpha)*r;
                                yf5(counter) = out.intMatrixY(n,m)-sind(alpha)*r;
                                xpi = out.intMatrixX(n,m)-cosd(alpha)*r;
                                ypi = out.intMatrixY(n,m)+sind(alpha)*r;
                                if flagfinal==1
                                    xf5(counter) = out.intMatrixX(n,m);
                                    yf5(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            else
                                if flaginiti==1 & flagpass==0
                                    flagpass = 1;
                                    xf5(counter) = out.intMatrixX(n,m);
                                    yf5(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            end
                            xpi_end = xpi;
                            ypi_end = ypi;
                        end
                    end
                end
                xi5(counter+1) = xpi_end;
                yi5(counter+1) = ypi_end;
                xf5(counter+1) = xf(n);
                yf5(counter+1) = yf(n);
                counter = 0;
            end
            M = [M; sortrows([xi5' yi5'; xf5' yf5'])];
            xi5 = 0*xi5; yi5 = 0*yi5; xf5 = 0*xf5; yf5 = 0*yf5;
        end
        counter = 0;
        for n=1:length(M)
            if sum(M(n,:))~=0
                counter = counter+1;
                M5(counter,:) = M(n,:);
            end
        end

        %mapeamento dos segmentos

        counter = 0;
        M = zeros(1,2);
        xi7 = 0; yi7 = 0; xf7 = 0; yf7 = 0;
        for n=1:length(xi)
            if sum(out.intAdjacencyMatrix(n,:))>=1
                xpi = xi(n);
                ypi = yi(n);
                for m=1:length(xi)
                    flagpass = 0;
                    if out.intAdjacencyMatrix(n,m)==1
                        flaginiti = 0;
                        flagfinal = 0;
                        counter = counter+1;
                        xi7(counter) = xpi;
                        yi7(counter) = ypi;
                        if xi(n)==out.intMatrixX(n,m) & yi(n)==out.intMatrixY(n,m)
                            flaginiti = 1;
                        end
                        if xf(n)==out.intMatrixX(n,m) & yf(n)==out.intMatrixY(n,m)
                            flagfinal = 1;
                        end
                        if xpi<=out.intMatrixX(n,m) & ypi<=out.intMatrixY(n,m)
                            if flaginiti==0
                                xf7(counter) = out.intMatrixX(n,m);
                                yf7(counter) = out.intMatrixY(n,m);
                                xpi = out.intMatrixX(n,m);
                                ypi = out.intMatrixY(n,m);
                                if flagfinal==1
                                    xf7(counter) = out.intMatrixX(n,m);
                                    yf7(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            else
                                if flaginiti==1 & flagpass==0
                                    flagpass = 1;
                                    xf7(counter) = out.intMatrixX(n,m);
                                    yf7(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            end
                            xpi_end = xpi;
                            ypi_end = ypi;
                        end
                        if xpi>=out.intMatrixX(n,m) & ypi>=out.intMatrixY(n,m)
                            if flaginiti==0
                                xf7(counter) = out.intMatrixX(n,m);
                                yf7(counter) = out.intMatrixY(n,m);
                                xpi = out.intMatrixX(n,m);
                                ypi = out.intMatrixY(n,m);
                                if flagfinal==1
                                    xf7(counter) = out.intMatrixX(n,m);
                                    yf7(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            else
                                if flaginiti==1 & flagpass==0
                                    flagpass = 1;
                                    xf7(counter) = out.intMatrixX(n,m);
                                    yf7(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            end
                            xpi_end = xpi;
                            ypi_end = ypi;
                        end
                        if xpi<=out.intMatrixX(n,m) & ypi>=out.intMatrixY(n,m)
                            if flaginiti==0
                                xf7(counter) = out.intMatrixX(n,m);
                                yf7(counter) = out.intMatrixY(n,m);
                                xpi = out.intMatrixX(n,m);
                                ypi = out.intMatrixY(n,m);
                                if flagfinal==1
                                    xf7(counter) = out.intMatrixX(n,m);
                                    yf7(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            else
                                if flaginiti==1 & flagpass==0
                                    flagpass = 1;
                                    xf7(counter) = out.intMatrixX(n,m);
                                    yf7(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            end
                            xpi_end = xpi;
                            ypi_end = ypi;
                        end
                        if xpi>=out.intMatrixX(n,m) & ypi<=out.intMatrixY(n,m)
                            if flaginiti==0
                                xf7(counter) = out.intMatrixX(n,m);
                                yf7(counter) = out.intMatrixY(n,m);
                                xpi = out.intMatrixX(n,m);
                                ypi = out.intMatrixY(n,m);
                                if flagfinal==1
                                    xf7(counter) = out.intMatrixX(n,m);
                                    yf7(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            else
                                if flaginiti==1 & flagpass==0
                                    flagpass = 1;
                                    xf7(counter) = out.intMatrixX(n,m);
                                    yf7(counter) = out.intMatrixY(n,m);
                                    xpi = out.intMatrixX(n,m);
                                    ypi = out.intMatrixY(n,m);
                                end
                            end
                            xpi_end = xpi;
                            ypi_end = ypi;
                        end
                    end
                end
                xi7(counter+1) = xpi_end;
                yi7(counter+1) = ypi_end;
                xf7(counter+1) = xf(n);
                yf7(counter+1) = yf(n);
                counter = 0;
            end
            M = [M; sortrows([xi7' yi7'; xf7' yf7'])];
            xi7 = 0*xi7; yi7 = 0*yi7; xf7 = 0*xf7; yf7 = 0*yf7;
        end
        counter = 0;
        for n=1:length(M)
            if sum(M(n,:))~=0
                counter = counter+1;
                M7(counter,:) = M(n,:);
            end
        end
        if flagin==1 & flagni==1
            M2plot = [xi2' yi2' xf2' yf2'];
            counter = 0;
            for n=1:2:length(M7)
                counter = counter+1;
                M7p(counter,:) = [M7(n,1) M7(n,2) M7(n+1,1) M7(n+1,2)];
            end
            Mp = [M2plot; M7p];
        end
        if flagin==1 & flagni==0
            counter = 0;
            for n=1:2:length(M7)
                counter = counter+1;
                M7p(counter,:) = [M7(n,1) M7(n,2) M7(n+1,1) M7(n+1,2)];
            end
            Mp = [M7p];
        end
        if flagin==0
            Mp = [xi yi xf yf];
        end
        dim_Mp = size(Mp);
        counter = 0;
        for n=1:dim_Mp
            if Mp(n,1)==Mp(n,3) & Mp(n,2)==Mp(n,4)
            else
                counter = counter+1;
                Mpdob(counter,:) = Mp(n,:);
            end
        end
        Mp = Mpdob;
        dim_Mp = size(Mp);

        %discriminacao I e X

        n_typex = 0;
        if flagin==1 & n_typey~=0
            typex = setdiff([xp' yp'],[x6y' y6y'],'rows','stable');
            x6x = typex(:,1);
            y6x = typex(:,2);
            n_typex = length(x6x);
        else
            if flagin==1
                x6x = xp;
                y6x = yp;
                n_typex = length(x6x);
            end
        end
        n_typei = 0;
        if n_typey~=0
            typei = setdiff([xi yi; xf yf],[x6y' y6y'],'rows','stable');
            x6i = typei(:,1);
            y6i = typei(:,2);
            n_typei = length(x6i);
        else
            if flagni==1
                x6i = [xi xf];
                y6i = [yi yf];
                n_typei = length(x6i);
            end
            if flagin==1
                typei = setdiff([xi yi; xf yf],[xp' yp'],'rows','stable');
                x6i = typei(:,1);
                y6i = typei(:,2);
                n_typei = length(x6i);
            end
        end

        %calculo do diagrama triangular I-Y-X

        ptypei = n_typei/(n_typei+n_typey+n_typex);
        ptypey = n_typey/(n_typei+n_typey+n_typex);
        ptypex = n_typex/(n_typei+n_typey+n_typex);
        Tx = cosd(60)-ptypey*cosd(60)+ptypex/2;
        Ty = sind(60)-ptypey*sind(60)-ptypex*cotd(30)/2;

        %discriminacao I-I, I-C e C-C

        n_typeii = 0;
        n_typeic = 0;
        n_typecc = 0;
        for n=1:length(Mp)
            flaginti(n) = 0;
            flagintf(n) = 0;
            for m=1:length(xp)
                if (Mp(n,1)==xp(m) & Mp(n,2)==yp(m))
                    flaginti(n) = 1;
                end
                if (Mp(n,3)==xp(m) & Mp(n,4)==yp(m))
                    flagintf(n) = 1;
                end
            end
            if (flaginti(n)==0 & flagintf(n)==0)
                n_typeii = n_typeii+1;
            end
            if (flaginti(n)==0 & flagintf(n)==1) | (flaginti(n)==1 & flagintf(n)==0)
                n_typeic = n_typeic+1;
            end
            if (flaginti(n)==1 & flagintf(n)==1)
                n_typecc = n_typecc+1;
            end
        end

        %calculo do diagrama triangular II-IC-CC

        ptypeii = n_typeii/(n_typeii+n_typeic+n_typecc);
        ptypeic = n_typeic/(n_typeii+n_typeic+n_typecc);
        ptypecc = n_typecc/(n_typeii+n_typeic+n_typecc);
        Txp = cosd(60)-ptypeic*cosd(60)+ptypecc/2;
        Typ = sind(60)-ptypeic*sind(60)-ptypecc*cotd(30)/2;

        %divisao dos segmentos que pertencem a regularizacao

        flagi2 = 1;
        counter = 0;
        for n=1:length(xi)
            if sum(out.intAdjacencyMatrix(n,:))>=2
                xpi = xi(n);
                ypi = yi(n);
                xi8(counter+1) = xpi;
                yi8(counter+1) = ypi;
                xf8(counter+1) = xf(n);
                yf8(counter+1) = yf(n);
                counter = length(xi8);
            end
        end
        if counter==0
            flagi2 = 0;
            fprintf('não há lineamentos que se intersectam ou se tocam mais de uma vez!\n\n');
        end

        %calculo do tensor de permeabilidade i

        if  flagi2==1;
            A = (max([xi' xf'])-min([xi' xf']))*(max([yi' yf'])-min([yi' yf']));
            for n=1:length(xi8)
                rk(n) = sqrt((xf8(n)-xi8(n))^2+(yf8(n)-yi8(n))^2);
                tk(n) = 1e-04*rk(n);%adequar aos dados!
                thetak(n) = atand((yf8(n)-yi8(n))/(xf8(n)-xi8(n)));
            end
            P11 = (1/A)*sum(rk.^2.*tk.^3.*sind(thetak).^2);
            P12 = (1/A)*sum(rk.^2.*tk.^3.*cosd(thetak).*sind(thetak));
            P21 = P12;
            P22 = (1/A)*sum(rk.^2.*tk.^3.*cosd(thetak).^2);
            k11 = (1/12)*(P11+P22-P11);
            k12 = (1/12)*(P12-0);
            k21 = (1/12)*(P21-0);
            k22 = (1/12)*(P11+P22-P22);
            [kvec,kval] = eig([k11 k12; k21 k22]);
        end

        %calculo do tensor de permeabilidade ii

        f = max(0, (2.94*(4*n_typex+2*n_typey))/(4*n_typex+2*n_typey+n_typei)-2.13);
        A = (max([xi' xf'])-min([xi' xf']))*(max([yi' yf'])-min([yi' yf']));
        for n=1:length(xi)
            rk(n) = sqrt((xf(n)-xi(n))^2+(yf(n)-yi(n))^2);
            tk(n) = 1e-04*rk(n);%adequar aos dados!
            thetak(n) = atand((yf(n)-yi(n))/(xf(n)-xi(n)));
        end
        P11 = (1/A)*sum(rk.^2.*tk.^3.*sind(thetak).^2);
        P12 = (1/A)*sum(rk.^2.*tk.^3.*cosd(thetak).*sind(thetak));
        P21 = P12;
        P22 = (1/A)*sum(rk.^2.*tk.^3.*cosd(thetak).^2);
        k211 = (1/12)*(P11+P22-P11)*f;
        k212 = (1/12)*(P12-0)*f;
        k221 = (1/12)*(P21-0)*f;
        k222 = (1/12)*(P11+P22-P22)*f;
        [k2vec,k2val] = eig([k211 k212; k221 k222]);

        %calculo do percentual relativo ao fator de aproximacao

        fprintf('percentual do fator de aproximação:\n\n');

        100*(factor/sqrt((max(max([xi xf]))-min(min([yi yf])))^2+(max(max([yi yf]))-min(min([yi yf])))^2))

        %calculo do CB

        xiprim = xi;
        yiprim = yi;
        xfprim = xf;
        yfprim = yf;
        xiarealim = (min(min([xiprim xfprim])));
        xfarealim = (max(max([xiprim xfprim])));
        yiarealim = (min(min([yiprim yfprim])));
        yfarealim = (max(max([yiprim yfprim])));
        prolatlon = (yfarealim-yiarealim)/(xfarealim-xiarealim);
        lanco = (xfarealim-xiarealim)/lados;
        for mm=round(lados*prolatlon):-1:1
            for nn=1:lados
                xilim = xiarealim+(nn-1)*lanco;
                xflim = xiarealim+nn*lanco;
                yilim = yiarealim+(mm-1)*lanco;
                yflim = yiarealim+mm*lanco;
                lquad = [xilim yilim xflim yilim;...
                    xflim yilim xflim yflim;...
                    xflim yflim xilim yflim;...
                    xilim yflim xilim yilim];
                xinfequad = (min(min([lquad(:,1) lquad(:,3)])));
                yinfequad = (min(min([lquad(:,2) lquad(:,4)])));
                xsupdquad = (max(max([lquad(:,1) lquad(:,3)])));
                ysupdquad = (max(max([lquad(:,2) lquad(:,4)])));
                xmedio = (xilim+xflim)/2;
                ymedio = (yilim+yflim)/2;
                latvec(mm) = ymedio;
                lonvec(nn) = xmedio;
                n_typei = 0;
                for n=1:length(x6i)
                    if x6i(n)>=xinfequad && x6i(n)<=xsupdquad && y6i(n)>=yinfequad && y6i(n)<=ysupdquad
                        n_typei = n_typei+1;
                    end
                end
                n_typey = 0;
                for n=1:length(x6y)
                    if x6y(n)>=xinfequad && x6y(n)<=xsupdquad && y6y(n)>=yinfequad && y6y(n)<=ysupdquad
                        n_typey = n_typey+1;
                    end
                end
                n_typex = 0;
                for n=1:length(x6x)
                    if x6x(n)>=xinfequad && x6x(n)<=xsupdquad && y6x(n)>=yinfequad && y6x(n)<=ysupdquad
                        n_typex = n_typex+1;
                    end
                end
                if n_typei~=0 || n_typey~=0 || n_typex~=0
                    CB(mm,nn) = (3*n_typey+4*n_typex)/((1/2)*(n_typei+3*n_typey+4*n_typex));
                else CB(mm,nn)=0;
                end
            end
        end

        %confeccao dos diagramas

        axes(ax{1,1});
        hold on
        for n=1:length(xi)
            plot([xi(n),xf(n)],[yi(n),yf(n)],'k');
        end
        axis([min(min([xi,xf])) max(max([xi,xf])) min(min([yi,yf])) max(max([yi,yf]))]);
        axis('equal');
        plot(x6i,y6i,'o','MarkerFaceColor','g','MarkerEdgeColor','g');
        if n_typey~=0
            plot(x6y,y6y,'^','MarkerFaceColor','r','MarkerEdgeColor','r');
        end
        if flagin==1
            plot(x6x,y6x,'s','MarkerFaceColor','b','MarkerEdgeColor','b');
        end
        %axis([min(min([xi,xf])) max(max([xi,xf])) min(min([yi,yf])) max(max([yi,yf]))]);
        axis('tight');
        axis('equal');
        title('nós I,Y e X');
        axes(ax{1,2});
        hold on
        Mnosgeol = [15.805556 27.805556 93.000000 162.500000;
                    93.000000 162.500000 107.250000 126.027778;
                    107.250000 126.027778 111.944444 85.222222;
                    111.944444 85.222222 92.444444 48.750000;
                    92.444444 48.750000 15.805556 27.805556];
        Mnosgeol(:,1) = Mnosgeol(:,1)/190;
        Mnosgeol(:,2) = Mnosgeol(:,2)/190;
        Mnosgeol(:,3) = Mnosgeol(:,3)/190;
        Mnosgeol(:,4) = Mnosgeol(:,4)/190;
        Mnostect = [117.348688 197.476647 185.069738 207.298784
                    95.636596 161.806782 204.197057 175.764555];
        Mnostect(:,1) = Mnostect(:,1)/308;
        Mnostect(:,2) = Mnostect(:,2)/308;
        Mnostect(:,3) = Mnostect(:,3)/308;
        Mnostect(:,4) = Mnostect(:,4)/308;
        for n=1:5
            plot([Mnosgeol(n,1) Mnosgeol(n,3)],[Mnosgeol(n,2) Mnosgeol(n,4)],'m','LineWidth',2);
        end
        for n=1:2
            plot([Mnostect(n,1) Mnostect(n,3)],[Mnostect(n,2) Mnostect(n,4)],'--k','LineWidth',2);
        end
        plot([0,0.5,1,0],[0,0.866,0,0],'k','LineWidth',3);
        plot(Tx,Ty,'.r','MarkerSize',25);
        axis([-0.1 1.1 -0.1 1]);
        axis('equal');
        axis off;
        text(0.5,0.95,'I','FontWeight','bold');
        text(-0.1,0,'Y','FontWeight','bold');
        text(1.05,0,'X','FontWeight','bold');
        title('proporções I,Y e X');
        axes(ax{2,1});
        hold on
        for n=1:dim_Mp(1,1)
            if (flaginti(n)==0 & flagintf(n)==0)
                plot([Mp(n,1),Mp(n,3)],[Mp(n,2),Mp(n,4)],'g');
            end
            if (flaginti(n)==0 & flagintf(n)==1) | (flaginti(n)==1 & flagintf(n)==0)
                plot([Mp(n,1),Mp(n,3)],[Mp(n,2),Mp(n,4)],'r');
            end
            if (flaginti(n)==1 & flagintf(n)==1)
                plot([Mp(n,1),Mp(n,3)],[Mp(n,2),Mp(n,4)],'b');
            end
        end
        %axis([min(min([xi,xf])) max(max([xi,xf])) min(min([yi,yf])) max(max([yi,yf]))]);
        axis('tight');
        axis('equal');
        title('segmentos I-I,I-C e C-C');
        axes(ax{2,2});
        hold on
        Mseggeol = [29.083333 49.472222 85.416667 148.416667;
                    85.416667 148.416667 93.527778 136.861111;
                    93.527778 136.861111 85.222222 81.611111;
                    84.861111 81.250000 143.361111 14.805556;
                    143.361111 14.805556 137.944444 0.000000;
                    137.944444 0.000000 66.083333 0.361111;
                    66.083333 0.361111 29.083333 49.472222];
        Mseggeol(:,1) = Mseggeol(:,1)/190;
        Mseggeol(:,2) = Mseggeol(:,2)/190;
        Mseggeol(:,3) = Mseggeol(:,3)/190;
        Mseggeol(:,4) = Mseggeol(:,4)/190;
        Msegtect = [156.120282 262.095969 143.196417 240.383877
                    143.196417 240.383877 130.789507 215.570058
                    130.789507 215.570058 121.484325 192.307102
                    121.484325 192.307102 114.246961 170.595010
                    113.730006 170.078055 107.594370 149.916827
                    107.594370 149.916827 105.458733 131.823417
                    105.458733 131.823417 104.941779 112.179143
                    104.941779 112.179143 106.492642 97.187460
                    106.492642 97.187460 111.662188 81.678823
                    111.662188 81.678823 117.348688 67.721049
                    117.348688 67.721049 127.170825 54.280230
                    127.170825 54.280230 138.543826 42.390275
                    138.543826 42.390275 150.433781 32.051184
                    150.433781 32.051184 167.493282 23.779910
                    166.976328 23.779910 185.586692 16.542546
                    185.586692 16.542546 205.747921 10.339091
                    205.747921 10.339091 228.493922 5.169546
                    228.493922 5.169546 254.341651 3.101727
                    254.341651 3.101727 281.740243 0.516955
                    281.740243 0.516955 310.689699 -1.033909];
        Msegtect(:,1) = Msegtect(:,1)/308;
        Msegtect(:,2) = Msegtect(:,2)/308;
        Msegtect(:,3) = Msegtect(:,3)/308;
        Msegtect(:,4) = Msegtect(:,4)/308;
        for n=1:7
            plot([Mseggeol(n,1) Mseggeol(n,3)],[Mseggeol(n,2) Mseggeol(n,4)],'m','LineWidth',2);
        end
        for n=1:20
            plot([Msegtect(n,1) Msegtect(n,3)],[Msegtect(n,2) Msegtect(n,4)],'--k','LineWidth',2);
        end
        plot(Msegtect(10,3),Msegtect(10,4),'.k','MarkerSize',20);
        plot(Msegtect(13,3),Msegtect(13,4),'.k','MarkerSize',20);
        plot([0,0.5,1,0],[0,0.866,0,0],'k','LineWidth',3);
        plot(Txp,Typ,'.r','MarkerSize',25);
        axis([-0.1 1.1 -0.1 1]);
        axis('equal');
        axis off;
        text(0.45,0.95,'I-I','FontWeight','bold');
        text(-0.18,0,'I-C','FontWeight','bold');
        text(1.05,0,'C-C','FontWeight','bold');
        title('proporções I-I,I-C e C-C');
        if flagi2==1
            axes(ax{3,1});
            hold on
            kvalnorm = kval/min([kval(1,1) kval(2,2)]);
            plot(0,0,'.k');
            plot([0 kvec(1,1)*kvalnorm(1,1)],[0 kvec(1,2)*kvalnorm(1,1)],'k');
            plot([0 -kvec(1,1)*kvalnorm(1,1)],[0 -kvec(1,2)*kvalnorm(1,1)],'k');
            plot([0 kvec(1,2)*kvalnorm(2,2)],[0 kvec(2,2)*kvalnorm(2,2)],'k');
            plot([0 -kvec(1,2)*kvalnorm(2,2)],[0 -kvec(2,2)*kvalnorm(2,2)],'k');
            axis('tight');
            axis('equal');
            text(1,0.5*max([kvec(1,2)*kvalnorm(1,1) -kvec(1,2)*kvalnorm(1,1) kvec(2,2)*kvalnorm(2,2) -kvec(2,2)*kvalnorm(2,2)]),num2str(max(max(kvalnorm))));
            title('razão de permeabilidade i');
        end
        axes(ax{3,2});
        hold on
        k2valnorm = k2val/min([k2val(1,1) k2val(2,2)]);
        plot(0,0,'.k');
        plot([0 k2vec(1,1)*k2valnorm(1,1)],[0 k2vec(1,2)*k2valnorm(1,1)],'k');
        plot([0 -k2vec(1,1)*k2valnorm(1,1)],[0 -k2vec(1,2)*k2valnorm(1,1)],'k');
        plot([0 k2vec(1,2)*k2valnorm(2,2)],[0 k2vec(2,2)*k2valnorm(2,2)],'k');
        plot([0 -k2vec(1,2)*k2valnorm(2,2)],[0 -k2vec(2,2)*k2valnorm(2,2)],'k');
        axis('tight');
        axis('equal');
        text(1,0.5*max([k2vec(1,2)*k2valnorm(1,1) -k2vec(1,2)*k2valnorm(1,1) k2vec(2,2)*k2valnorm(2,2) -k2vec(2,2)*k2valnorm(2,2)]),num2str(max(max(k2valnorm))));
        text(1,0.4*max([k2vec(1,2)*k2valnorm(1,1) -k2vec(1,2)*k2valnorm(1,1) k2vec(2,2)*k2valnorm(2,2) -k2vec(2,2)*k2valnorm(2,2)]),num2str(f));
        title('razão de permeabilidade ii');
        axes(ax{4,1});
        hold on;
        resolucao = 100;
        loni = linspace(min(min([xiprim,xfprim])),max(max([xiprim,xfprim])),resolucao)';
        lati = linspace(min(min([yiprim,yfprim])),max(max([yiprim,yfprim])),resolucao)';
        [lonx,laty] = meshgrid(loni,lati);
        CBint = griddata(lonvec,latvec,CB,lonx,laty,'v4');
        for m=1:length(lonx)
            for n=1:length(laty)
                if CBint(m,n)>max(max(CB))
                    CBint(m,n) = max(max(CB));
                end
                if CBint(m,n)<min(min(CB))
                    CBint(m,n) = min(min(CB));
                end
            end
        end
        contourf(lonx,laty,CBint);
        colorbar('EastOutside');
        axis([min(min([xiprim,xfprim])) max(max([xiprim,xfprim])) min(min([yiprim,yfprim])) max(max([yiprim,yfprim]))]);
        axis('equal');
        title('C_{B}');
        axes(ax{4,2});
        hold on
        for n=1:length(xi)
            plot([xi(n),xf(n)],[yi(n),yf(n)],'k');
        end
        %axis([min(min([xi,xf])) max(max([xi,xf])) min(min([yi,yf])) max(max([yi,yf]))]);
        axis('tight');
        axis('equal');
        title('todos');

    end

fprintf('veja o diagrama para visualizar os resultados\n\n');

fprintf('concluído\n\n');

end