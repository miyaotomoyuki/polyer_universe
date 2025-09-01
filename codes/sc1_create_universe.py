import pickle
import glob
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from chemistry.mol_curation import CurateMolsMT, CurateMol, CurateMolsSDF
from chemistry.visualization import WriteDataFrameSmilesToXls
from chemistry.descriptors import calcDescriptorSet, rdkit_descriptorlist
from util.utility import IsMac
from pathlib import Path
import logging 
import pandas as pd
from util.utility import MakeFolderWithCurrentFuncName
from joblib import Parallel, cpu_count, delayed

logger=logging.getLogger(__name__)

def sc1CurateDB1(fd):
    outfd = MakeFolderWithCurrentFuncName(f'{fd}/results', allow_override=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'{outfd}/logging_sc1.txt', mode='w'),
            logging.StreamHandler()
        ]
    )

    # DB1 curation
    db1File = '/Users/miyao/work/collaboration/academia/Prof-Mase/Database/Reagent_Dataset.csv'
    problematic_rows = []
    db1 = pd.read_csv(db1File, sep=',', index_col=0, encoding='shift-jis')
    logger.info(f'Loaded recrods: {len(db1)}')
    db1['ROMol'] = db1['IsomericSMILES'].apply(Chem.MolFromSmiles)
    assert db1['ROMol'].isna().sum() == 0

    
    # curation 
    db1['neutral_SMILES_mix'] = db1['ROMol'].apply(lambda x: CurateMol(x, extract_biggest=False, return_smiles=True, remove_exclude=False))
    db1['neutral_SMILES_biggest'] = db1['ROMol'].apply(lambda x: CurateMol(x, extract_biggest=True, return_smiles=True, remove_exclude=False))
    db1_nonna = db1[~db1['neutral_SMILES_biggest'].isna()]
    logger.info(f'After curateion (only extracting the neutralalized biggest components): {len(db1_nonna)}')
    db1_nonna.to_csv(f'{outfd}/nonna_db1_{len(db1_nonna)}records.tsv', sep='\t')
    udb1_single = db1_nonna.drop_duplicates(subset='neutral_SMILES_biggest', keep='first')
    logger.info(f'Removing duplicate recrods basd on "single nutral SMILES component: {len(udb1_single)}')
    udb1_single.to_csv(f'{outfd}/nonna_unique_db1_{len(udb1_single)}records.tsv', sep='\t')

def sc2ExtractMethacrylate(fd):  
    print(rdkit_descriptorlist)   
    outfd = MakeFolderWithCurrentFuncName(f'{fd}/results', allow_override=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'{outfd}/logging_sc2.txt', mode='w'),
            logging.StreamHandler()
        ]
    )
    methacrylate_query  = '[CH2]=C([CH3])[CX3](=O)[OX2]'
    acrylate_query      = '[CH2]=[CH][CX3](=O)[OX2]'
    vinyl_query         = '[CH2]=[CH]'
    
    qmethacrylate       = Chem.MolFromSmarts(methacrylate_query)
    qacrylate           = Chem.MolFromSmarts(acrylate_query)
    qvinyl              = Chem.MolFromSmarts(vinyl_query)

    logger.info(f'SMARTS_QUERY: {methacrylate_query}')
    curatedMols = pd.read_csv(f'{fd}/results/sc1CurateDB1/nonna_unique_db1_35992records.tsv',sep='\t', index_col=0)
    logger.info(f'Loaded mols: {len(curatedMols)}')
    
    curatedMols['ROMol'] = curatedMols['neutral_SMILES_biggest'].apply(Chem.MolFromSmiles)
    curatedMols['num_methacrylates'] = curatedMols['ROMol'].apply(lambda x: len(x.GetSubstructMatches(qmethacrylate)))
    methacrylate = curatedMols[curatedMols['num_methacrylates']>0]
    methacrylate['num_acrylates'] = curatedMols['ROMol'].apply(lambda x: len(x.GetSubstructMatches(qacrylate)))
    methacrylate['num_vinyls']    = curatedMols['ROMol'].apply(lambda x: len(x.GetSubstructMatches(qvinyl)))
     
    logger.info(f'methacrylates: {len(methacrylate)}')
    outfname    = f'{outfd}/db1_methacrylates_{len(methacrylate)}'

    # calculate MW, logP, # num acceptors, # num donors
    desclist = ['MolWt', 'MolLogP', 'NumHAcceptors','NumHDonors','NumRotatableBonds', 'RingCount','NumAromaticRings','FractionCSP3']
    descs       = calcDescriptorSet(methacrylate['ROMol'], use_descriptors=desclist, input_mol=True)
    methacrylate = methacrylate.join(descs)

    
    methacrylate.to_csv(outfname + '.tsv', sep='\t')
    WriteDataFrameSmilesToXls(methacrylate, 
                              smiles_colnames=['IsomericSMILES', 'neutral_SMILES_mix','neutral_SMILES_biggest'], 
                              retain_smiles_col=True, 
                              out_filename=outfname + '.xlsx',
                              cell_width=300, 
                              cell_height=300)

def sc3CurateACD(fd, debug=True):
    outfd   = MakeFolderWithCurrentFuncName(f'{fd}/results', allow_override=True)
    if IsMac():
        acdfd   = '/Users/miyao/work/datasets/ACD/2025/BIOVIA_Content_2025.acd202501_2dsdf'
    else:
        acdfd   = '/home/miyao/work/datasets/acd2025'
    sdlist  = glob.glob(f'{acdfd}/*.sdf')
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'{outfd}/logging_sc3.txt', mode='w'),
            logging.StreamHandler()
        ]
    )

    # multiprocessing 
    njobs   = cpu_count() -1 if not debug else 2
    batches = np.array_split(sdlist, njobs)    
    _ = Parallel(n_jobs=njobs)(delayed(worker_curation)(batch, outfd, idx) for idx, batch in enumerate(batches))
    

def worker_curation(sdfiles, outfd, id=0):
    for sdfile in sdfiles:
        sdfpath = Path(sdfile)
        logger.info(f'------start to processing: {sdfpath.name}')
        outsmiles = str(sdfpath.stem) + '.smi'
        outpath   = f'{outfd}/{outsmiles}'
        progress=True if id == 0 else False
        CurateMolsSDF(sdfile, outpath, verbose=False, show_progress=progress)


def sc4PostProcessACDs(fd, debug=True):
    outfd   = MakeFolderWithCurrentFuncName(f'{fd}/results', allow_override=True)
    cfileList = glob.glob(f'{fd}/results/sc3CurateACD/*[0-9].tsv')
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'{outfd}/logging_sc4.txt', mode='w'),
            logging.StreamHandler()
        ]
    )
    outmols = pd.read_csv(cfileList[0], sep='\t') # first element as a template
    outmols.drop(columns='Name', inplace=True)
    for idx, fname in enumerate(cfileList[1:]):
        if debug and idx > 3:
            break
        logger.info(f'processing: {fname}')
        loaded = pd.read_csv(fname, sep='\t')
        loaded.drop(columns='Name', inplace=True)
        outmols = pd.concat([outmols, loaded], ignore_index=True)
    
    logger.info(f'loaded all mols. {len(outmols)}')
    uniquemols = outmols.drop_duplicates(subset='neutral_SMILES_biggest', keep='first')
    logger.info(f'loaded unique mols. {len(uniquemols)}')

    # save the unique mols 
    outmols.to_csv(f'{outfd}/curated_allmols_ACD_{len(outmols)}.tsv', sep='\t')
    uniquemols.to_csv(f'{outfd}/unique_mols_inBiggestNeutral_{len(uniquemols)}.tsv', sep='\t')

    # convert to rdmol and save as pickle
    logger.info('--------converting to ROMol---------')
    uniquemols['ROMol'] = uniquemols['neutral_SMILES_biggest'].apply(Chem.MolFromSmiles)
    okuniquemols       = uniquemols[~uniquemols['ROMol'].isna()]
    logger.info(f'After dropping mols failing to conver to ROMol: {len(okuniquemols)}')
    resname            = f'unique_mols_inBiggestNeutral_rdkit_convertible{len(okuniquemols)}'
    okuniquemols.to_csv(f'{outfd}/{resname}.tsv', sep='\t')
    pickle.dump(okuniquemols, open(f'{outfd}/{resname}.pickle','wb'))
    
    
def test_smarts_qurey():
    smarts = '[N+0$([NH]([CX4,c])),N+0$([N]([CX4,c])([CX4,c])):1]-[C$(C(=O)([CX4,c])),C$([CH0](=O)):2]=[O:3]>>[Cl,OH,O:4][C:2]=[O:3].[N:1]'
    smi = 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C'
    rxn = rdChemReactions.ReactionFromSmarts(smarts)
    imatinib = Chem.MolFromSmiles(smi)
    product = rxn.RunReactants((imatinib,))
    print(Chem.MolToSmiles(product[0][0]), Chem.MolToSmiles(product[0][1]))

def sc5SelectMethacrylateACD(fd):
    """
    The same procedure for sc2
    """
    outfd = MakeFolderWithCurrentFuncName(f'{fd}/results', allow_override=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'{outfd}/logging_sc2.txt', mode='w'),
            logging.StreamHandler()
        ]
    )

    methacrylate_query  = '[CH2]=C([CH3])[CX3](=O)[OX2]'
    acrylate_query      = '[CH2]=[CH][CX3](=O)[OX2]'
    vinyl_query         = '[CH2]=[CH]'
    
    qmethacrylate       = Chem.MolFromSmarts(methacrylate_query)
    qacrylate           = Chem.MolFromSmarts(acrylate_query)
    qvinyl              = Chem.MolFromSmarts(vinyl_query)
    curatedMols         = pickle.load(open(f'{fd}/results/sc4PostProcessACDs/unique_mols_inBiggestNeutral_rdkit_convertible15827703.pickle', 'rb'))

    curatedMols['num_methacrylates'] = curatedMols['ROMol'].apply(lambda x: len(x.GetSubstructMatches(qmethacrylate)))
    methacrylate = curatedMols[curatedMols['num_methacrylates']>0]
    methacrylate['num_acrylates'] = curatedMols['ROMol'].apply(lambda x: len(x.GetSubstructMatches(qacrylate)))
    methacrylate['num_vinyls']    = curatedMols['ROMol'].apply(lambda x: len(x.GetSubstructMatches(qvinyl)))
     
    logger.info(f'methacrylates: {len(methacrylate)}')
    outfname    = f'{outfd}/acd_methacrylates_{len(methacrylate)}'

    # calculate MW, logP, # num acceptors, # num donors
    desclist = ['MolWt', 'MolLogP', 'NumHAcceptors','NumHDonors','NumRotatableBonds', 'RingCount','NumAromaticRings','FractionCSP3']
    descs       = calcDescriptorSet(methacrylate['ROMol'], use_descriptors=desclist, input_mol=True)
    methacrylate = methacrylate.join(descs)
    
    methacrylate.to_csv(outfname + '.tsv', sep='\t')
    WriteDataFrameSmilesToXls(methacrylate, 
                              smiles_colnames=['SMILES', 'neutral_SMILES_biggest'], 
                              retain_smiles_col=True, 
                              out_filename=outfname + '.xlsx',
                              cell_width=300, 
                              cell_height=300)

if __name__ == '__main__':
    bf = Path(__file__).parents[1]
    if 0:
        sc1CurateDB1ACD(bf)
    if 0:
        sc2ExtractMethacrylate(bf)
    if 0:
        sc3CurateACD(bf, debug=False)
    if 0:
        sc4PostProcessACDs(bf, debug=False)
    if 1:
        sc5SelectMethacrylateACD(bf)