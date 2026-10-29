import tempfile
import unittest
from pathlib import Path
from vaultrelink import Encoder, chunks, find_broken, load_vault, report

class VaultTests(unittest.TestCase):
    def test_scan_and_no_source_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'source.md'
            text = '[[missing#section|caption]] [[existing]] ![[image.png]]'
            source.write_text(text, encoding='utf-8')
            (root / 'existing.md').write_text('hello', encoding='utf-8')
            notes, warnings = load_vault(root)
            broken = find_broken(notes)
            self.assertFalse(warnings)
            self.assertEqual([b['target'] for b in broken], ['missing'])
            self.assertEqual(source.read_text(encoding='utf-8'), text)

    def test_html_escaping_and_alias(self):
        item = {'source':'<script>.md', 'raw':'[[old#section|caption]]', 'line':1, 'context':'<b>unsafe</b>'}
        candidate = {'path':'new.md','excerpt':'<img src=x>', 'score':.5, 'cosine':.5}
        out = report([(item,[candidate])], [], 2)
        self.assertNotIn('<img src=x>', out)
        self.assertIn('[[new#section|caption]]', out)
        self.assertIn('&lt;script&gt;', out)

    def test_duplicate_names_are_not_misreported_as_missing(self):
        notes = [{'path':'x/source.md','stem':'source','text':'[[same]]'},
                 {'path':'x/same.md','stem':'same','text':''}, {'path':'y/same.md','stem':'same','text':''}]
        self.assertEqual(find_broken(notes), [])

    def test_long_note_chunks_include_end(self):
        encoder = Encoder()
        source = ('memory garden roses ' * 300) + ' uniquefinalmarker'
        pieces = chunks(source, encoder.tokenizer)
        self.assertGreater(len(pieces), 1)
        self.assertIn('uniquefinalmarker', pieces[-1])
        self.assertEqual(len(encoder.encode([pieces[0], pieces[-1]])), 2)

if __name__ == '__main__':
    unittest.main()
